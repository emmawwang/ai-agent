import pandas as pd
import numpy as np
import os
import requests
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from task_processor import extract_tasks, preprocess_text, detect_industry
from config import ECONOMIC_DATA_FILE, BLS_EMPLOYMENT_FILE, ONET_TASK_MAPPINGS_FILE, HIGH_GROWTH_THRESHOLD, MODERATE_GROWTH_THRESHOLD
from job_fetcher import fetch_job_from_url

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def analyze_job_automation(job_input, csv_path=None, similarity_threshold=0.15):
    """
    Analyze a job description for automation risk using a CSV of tasks with automation scores.
    
    Args:
        job_input: Either a job description string or a URL to a job posting
        csv_path: Path to the CSV file with automation data
        similarity_threshold: Threshold for task matching
        
    Returns:
        Dictionary containing analysis results
    """
    # Use the default ONET task mappings if no path provided
    if csv_path is None:
        csv_path = ONET_TASK_MAPPINGS_FILE
        
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Variable to store the original URL if provided
    job_url = None
    
    # If job_input is a URL, fetch the job description
    if isinstance(job_input, str) and (job_input.startswith("http://") or job_input.startswith("https://")):
        job_url = job_input
        try:
            job_description = fetch_job_from_url(job_input)
            # Also fetch company information in parallel
            company_info = get_company_information(job_url)
        except Exception as e:
            print(f"Error fetching job description from URL: {e}")
            print("Please provide the job description directly.")
            return {
                "error": f"Failed to fetch job description from URL: {str(e)}",
                "Overall_Automation_Risk": 0.0
            }
    else:
        job_description = job_input
        company_info = None
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tasks from {csv_path}")
    
    # Extract tasks from job description
    job_tasks = extract_tasks(job_description)
    print(f"Extracted {len(job_tasks)} tasks from job description")
    
    # Detect industry from job description
    industry = detect_industry(job_description)
    print(f"Detected industry: {industry}")
    
    # Find matching tasks in dataset
    matched_tasks = match_tasks_improved(job_tasks, df, similarity_threshold)
    print(f"Found {len(matched_tasks)} matching tasks with threshold {similarity_threshold}")
    
    if not matched_tasks:
        print("Warning: No matching tasks found in dataset")
        return {
            "error": "No matching tasks found",
            "Industry": industry,  # Still include the detected industry
            "Overall_Automation_Risk": 0.0,
            "Company_Info": company_info,
            "Career_Growth_Potential": get_career_insights_from_perplexity(company_info, industry),
            "Industry_Stability": get_fallback_industry_data(0.3, industry)["industry_stability"]
        }
    
    # Calculate overall risk (weighted by similarity and confidence)
    weights = [t['match_score'] for t in matched_tasks]
    risks = [t['automation_pct'] for t in matched_tasks]
    overall_risk = np.average(risks, weights=weights)
    
    # Get industry stability data
    if company_info:
        # Use Perplexity API to get more accurate career growth data
        career_growth = get_career_insights_from_perplexity(company_info, industry)
        industry_stability = get_fallback_industry_data(overall_risk, industry)["industry_stability"]
    else:
        # Fall back to general industry data
        industry_data = get_fallback_industry_data(overall_risk, industry)
        career_growth = industry_data["career_growth"]
        industry_stability = industry_data["industry_stability"]
    
    return {
        "Overall_Automation_Risk": overall_risk,
        "Industry": industry,
        "Company_Info": company_info,
        "Career_Growth_Potential": career_growth,
        "Industry_Stability": industry_stability,
        "Matched_Tasks": matched_tasks
    }

def match_tasks_improved(job_tasks, task_data, similarity_threshold=0.05):
    """Match job tasks to dataset tasks with improved accuracy and better fallbacks"""
    # Ensure task_data is loaded and not empty
    if task_data.empty:
        raise ValueError("Task data is empty")
    
    # Debug print column names
    print(f"CSV columns: {task_data.columns.tolist()}")
    
    # First, try to auto-detect columns based on common naming patterns
    task_col = None
    pct_col = None
    
    # Look for task name column
    task_name_candidates = ['task_name', 'task description', 'task_statement', 'statement', 'task', 'description']
    for col in task_data.columns:
        if col.lower() in task_name_candidates:
            task_col = col
            print(f"Using '{task_col}' as task description column")
            break
    
    if task_col is None:
        # Fall back to first column with "task" in its name
        possible_task_cols = [col for col in task_data.columns if 'task' in col.lower()]
        if possible_task_cols:
            task_col = possible_task_cols[0]
            print(f"Using '{task_col}' as task description column")
        else:
            # Last resort - use the first string column that's not too short
            for col in task_data.columns:
                if task_data[col].dtype == 'object':
                    # Sample the data to see if it looks like task descriptions
                    sample = task_data[col].head(5).astype(str)
                    if sample.str.len().mean() > 15:  # Reasonable task descriptions are longer than 15 chars
                        task_col = col
                        print(f"Using '{task_col}' as task description column (auto-detected)")
                        break
            
            if task_col is None:
                raise ValueError("Could not auto-detect task description column")
    
    # Look for automation percentage column
    pct_candidates = ['pct', 'automation', 'probability', 'risk', 'auto_risk', 'probability_computerization']
    for col in task_data.columns:
        if col.lower() in pct_candidates or any(c in col.lower() for c in pct_candidates):
            sample = task_data[col].head(5)
            # Check if values look like percentages or decimals between 0-1
            try:
                sample = pd.to_numeric(sample, errors='coerce')
                if not sample.isna().all():
                    pct_col = col
                    print(f"Using '{pct_col}' as automation percentage column")
                    break
            except:
                pass
    
    if pct_col is None:
        # Last resort - try to find a numeric column with values between 0-1 or 0-100
        for col in task_data.columns:
            try:
                if task_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    col_min = task_data[col].min()
                    col_max = task_data[col].max()
                    # Check if it's in range 0-1 or 0-100
                    if (0 <= col_min <= 1 and 0 <= col_max <= 1) or (0 <= col_min <= 100 and 0 <= col_max <= 100):
                        pct_col = col
                        print(f"Using '{pct_col}' as automation percentage column (auto-detected)")
                        break
            except:
                pass
        
        if pct_col is None:
            raise ValueError("Could not auto-detect automation percentage column")
    
    # Sample the data for debugging
    print("\nSample task data:")
    for i in range(min(3, len(task_data))):
        try:
            task_text = task_data.iloc[i][task_col]
            pct_value = task_data.iloc[i][pct_col]
            print(f"  {i+1}. Task: '{task_text[:70]}...' - Automation: {pct_value}")
        except:
            print(f"  Error displaying sample {i}")
    
    # Normalize and clean job tasks
    job_tasks_clean = [preprocess_text(task) for task in job_tasks if task and isinstance(task, str)]
    
    # Debug print tasks from job description
    print("\nTasks from job description:")
    for i, task in enumerate(job_tasks):
        print(f"  {i+1}. {task[:70]}..." if len(task) > 70 else f"  {i+1}. {task}")
    
    # If no job tasks were found after preprocessing, try splitting more aggressively
    if not job_tasks_clean:
        print("No tasks found after preprocessing, trying alternative extraction...")
        # Try to extract tasks from full text
        full_text = " ".join([t for t in job_tasks if isinstance(t, str)])
        
        # Extract sentences that look like tasks
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        job_tasks_clean = [
            preprocess_text(s) for s in sentences 
            if len(s.strip()) > 20 and re.search(r'\b(?:manage|develop|create|analyze|implement|design|organize|lead|conduct|ensure|provide|maintain|support|coordinate|build|prepare|research|evaluate|perform|work|communicate|collaborate)\b', s.lower())
        ]
        
        if job_tasks_clean:
            print(f"Extracted {len(job_tasks_clean)} tasks using alternative extraction")
            # Print the extracted tasks
            for i, task in enumerate(job_tasks_clean):
                print(f"  {i+1}. {task[:70]}..." if len(task) > 70 else f"  {i+1}. {task}")
    
    if not job_tasks_clean:
        print("WARNING: Could not extract any tasks from job description")
        # Create a generic task based on the entire description
        job_description_text = " ".join([t for t in job_tasks if isinstance(t, str)])
        job_tasks_clean = [preprocess_text(job_description_text[:500])]  # Use first 500 chars
        print(f"Using generic task: '{job_tasks_clean[0][:70]}...'")
    
    # Handle potential NaN values in dataset tasks
    dataset_tasks = task_data[task_col].fillna("").astype(str).apply(preprocess_text)
    
    # Provide more information about the dataset
    print(f"\nDataset contains {len(dataset_tasks)} tasks for matching")
    
    try:
        # More flexible vectorization 
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), 
            min_df=1, 
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Combine all tasks
        all_tasks = job_tasks_clean + dataset_tasks.tolist()
        
        # Check for empty tasks
        all_tasks = [task if task.strip() else "empty task placeholder" for task in all_tasks]
        
        tfidf_matrix = vectorizer.fit_transform(all_tasks)
        
        # Print some statistics
        print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Top features: {list(vectorizer.vocabulary_.keys())[:10]}")
        
    except ValueError as e:
        print(f"Error in vectorization: {e}")
        # Fallback to simpler vectorization
        print("Falling back to simpler vectorization")
        vectorizer = TfidfVectorizer(stop_words='english')
        all_tasks = job_tasks_clean + dataset_tasks.tolist()
        all_tasks = [task if task.strip() else "empty task placeholder" for task in all_tasks]
        tfidf_matrix = vectorizer.fit_transform(all_tasks)
    
    # Get vectors
    job_vectors = tfidf_matrix[:len(job_tasks_clean)]
    dataset_vectors = tfidf_matrix[len(job_tasks_clean):]
    
    # Find matches with improved scoring
    matches = []
    for i, task in enumerate(job_tasks_clean):
        # Skip empty tasks
        if not task.strip():
            continue
            
        # Get all similarities
        similarities = cosine_similarity(job_vectors[i:i+1], dataset_vectors)[0]
        
        # Print similarity statistics
        print(f"\nTask {i+1} similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
        
        # Get top 5 matches
        top_indices = similarities.argsort()[-5:][::-1]
        top_scores = [similarities[idx] for idx in top_indices]
        
        print(f"Top matches for task: '{task[:70]}...'")
        for j, (idx, score) in enumerate(zip(top_indices[:3], top_scores[:3])):
            print(f"  Match {j+1}: {score:.4f} - '{task_data.iloc[idx][task_col][:70]}...'")
        
        # Check if best match is good enough
        if len(top_indices) > 0 and similarities[top_indices[0]] >= similarity_threshold:
            best_similarity = similarities[top_indices[0]]
            
            # Extract automation percentage value
            pct_val = task_data.iloc[top_indices[0]][pct_col]
            
            # Handle different formats of percentage values
            if isinstance(pct_val, str) and '%' in pct_val:
                pct_val = float(pct_val.strip('%')) / 100
            elif isinstance(pct_val, (int, float)) and pct_val > 1:
                # If value is > 1, assume it's a percentage from 0-100
                pct_val = pct_val / 100
                
            matches.append({
                'job_task': job_tasks[i] if i < len(job_tasks) else task,
                'matched_task': task_data.iloc[top_indices[0]][task_col],
                'automation_pct': float(pct_val),
                'similarity': float(best_similarity),
                'match_score': float(best_similarity)
            })
            print(f"  ✓ Added match with score {best_similarity:.4f}")
        else:
            print(f"  ✗ No match found above threshold {similarity_threshold}")
    
    # If no matches found but we have job tasks, use the best match anyway
    if not matches and len(job_tasks_clean) > 0:
        print("\nNo matches above threshold. Using best available match as fallback.")
        i = 0  # Use first task
        similarities = cosine_similarity(job_vectors[i:i+1], dataset_vectors)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        # Extract automation percentage value with error handling
        try:
            pct_val = task_data.iloc[best_idx][pct_col]
            
            # Handle different formats of percentage values
            if isinstance(pct_val, str) and '%' in pct_val:
                pct_val = float(pct_val.strip('%')) / 100
            elif isinstance(pct_val, (int, float)) and pct_val > 1:
                # If value is > 1, assume it's a percentage from 0-100
                pct_val = pct_val / 100
                
            matches.append({
                'job_task': job_tasks[i] if i < len(job_tasks) else job_tasks_clean[i],
                'matched_task': task_data.iloc[best_idx][task_col],
                'automation_pct': float(pct_val),
                'similarity': float(best_score),
                'match_score': float(best_score)
            })
            print(f"  ✓ Added fallback match with score {best_score:.4f}")
        except Exception as e:
            print(f"  ✗ Error adding fallback match: {e}")
    
    return matches

def get_industry_data(industry, risk_level):
    """Get real industry data using Perplexity API with better sources"""
    if not PERPLEXITY_API_KEY:
        print("Warning: Perplexity API key not found. Using fallback industry data.")
        return get_fallback_industry_data(risk_level, industry)
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # More specific query with explicit source requirements
    query = (
        f"I need current, factual data about the {industry} industry job market from official sources like the Bureau of Labor Statistics, Occupational Outlook Handbook, or industry reports. "
        f"Please provide: "
        f"1) Current annual growth rate percentage with source, "
        f"2) 5-year job growth projection percentage with source, "
        f"3) Investment trend (growing/stable/declining) with evidence, "
        f"4) Risk of technological disruption (high/medium/low) with examples, "
        f"5) Top 3-5 most in-demand skills with source, "
        f"6) 1-2 specific career recommendations for professionals in this field. "
        f"Format as JSON with keys: growth_rate, projection, source, investment_trend, disruption_risk, in_demand_skills, career_recommendations. "
        f"Include only verified information from 2022-2024 sources. Include the specific URL or publication name for each data point."
    )
    
    # Perplexity API payload
    payload = {
        "model": "sonar",  # Using sonar model for web search capabilities
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.0,  # Zero temperature for maximum factuality
        "max_tokens": 1024
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, 
                               json=payload,
                               timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                industry_stats = json.loads(json_match.group(1))
                print("Successfully extracted industry data with sources")
            else:
                try:
                    # Try to parse the entire content as JSON
                    industry_stats = json.loads(content)
                    print("Successfully parsed industry data as JSON")
                except json.JSONDecodeError:
                    # Extract structured data from text response
                    extracted = extract_structured_data(content)
                    if extracted:
                        industry_stats = extracted
                        print("Extracted structured industry data from text")
                    else:
                        print("Using industry-specific fallback data")
                        return get_fallback_industry_data(risk_level, industry)
            
            # Process the API response into our required format
            return process_industry_api_data(industry_stats, industry, risk_level)
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return get_fallback_industry_data(risk_level, industry)
            
    except Exception as e:
        print(f"Error querying Perplexity API: {e}")
        return get_fallback_industry_data(risk_level, industry)

def extract_structured_data(text):
    """Extract structured data from text when JSON parsing fails"""
    try:
        data = {}
        
        # Look for growth rate
        growth_match = re.search(r'growth rate.*?(\d+\.?\d*%|\d+\.?\d*\s*percent)', text, re.IGNORECASE)
        if growth_match:
            data['growth_rate'] = growth_match.group(1)
            
        # Look for projection
        projection_match = re.search(r'projection.*?(\d+\.?\d*%|\d+\.?\d*\s*percent)', text, re.IGNORECASE)
        if projection_match:
            data['projection'] = projection_match.group(1)
            
        # Look for investment trend
        if 'growing' in text.lower():
            data['investment_trend'] = 'Growing'
        elif 'declining' in text.lower():
            data['investment_trend'] = 'Declining'
        else:
            data['investment_trend'] = 'Stable'
            
        # Look for disruption risk
        if 'high risk' in text.lower() or 'high disruption' in text.lower():
            data['disruption_risk'] = 'High'
        elif 'low risk' in text.lower() or 'low disruption' in text.lower():
            data['disruption_risk'] = 'Low'
        else:
            data['disruption_risk'] = 'Medium'
            
        # Extract skills (basic approach)
        skills = []
        skill_section = re.search(r'skills[:\-]*(.*?)(?:recommendations|\.|$)', text, re.IGNORECASE | re.DOTALL)
        if skill_section:
            skills_text = skill_section.group(1)
            # Extract skills separated by commas or listed with bullets/numbers
            skills = re.findall(r'(?:•|\*|-|\d+\.|,)\s*([A-Za-z][^,•\*\-\d\.]*)', skills_text)
            skills = [s.strip() for s in skills if s.strip()]
        
        data['in_demand_skills'] = skills if skills else ["Data not available"]
        
        # Extract recommendations (basic approach)
        recommendations = []
        rec_section = re.search(r'recommendations[:\-]*(.*?)(?:$|conclusion)', text, re.IGNORECASE | re.DOTALL)
        if rec_section:
            rec_text = rec_section.group(1)
            # Extract recommendations separated by periods or listed with bullets/numbers
            recommendations = re.findall(r'(?:•|\*|-|\d+\.)\s*([^.]*\.)', rec_text)
            recommendations = [r.strip() for r in recommendations if r.strip()]
        
        data['career_recommendations'] = recommendations if recommendations else ["Data not available"]
        
        return data
    except Exception as e:
        print(f"Error extracting structured data: {e}")
        return None

def process_industry_api_data(api_data, industry, risk_level):
    """Process API response into required format with source citations"""
    # Extract values with fallbacks
    growth_rate = api_data.get("growth_rate", "Unknown")
    projection = api_data.get("projection", "Unknown")
    source = api_data.get("source", "Recent industry analysis")
    investment_trend = api_data.get("investment_trend", "Stable")
    disruption_risk = api_data.get("disruption_risk", "Medium")
    in_demand_skills = api_data.get("in_demand_skills", ["Digital literacy", "Communication", "Problem solving"])
    career_recommendations = api_data.get("career_recommendations", ["Develop adaptable skills and continuous learning"])
    
    # Determine growth level based on projection with citations
    if isinstance(projection, str):
        if "%" in projection:
            try:
                projection_value = float(projection.strip("%"))
                if projection_value > 10:
                    growth_level = "High"
                elif projection_value > 5:
                    growth_level = "Moderate"
                else:
                    growth_level = "Low"
            except:
                growth_level = "Moderate"
        else:
            growth_level = "Moderate"
    else:
        growth_level = "Moderate"
    
    # Determine stability status
    if investment_trend and "growing" in investment_trend.lower():
        stability_status = "Growing"
    elif investment_trend and "declining" in investment_trend.lower():
        stability_status = "Declining"
    else:
        stability_status = "Stable"
    
    # Create career growth output with source citation
    career_growth = {
        "level": growth_level,
        "description": f"Based on {source}",
        "outlook": f"Projected growth: {projection}",
        "skill_demand": f"In-demand skills: {', '.join(in_demand_skills[:3]) if isinstance(in_demand_skills, list) else in_demand_skills}",
        "recommendations": f"{career_recommendations[0] if isinstance(career_recommendations, list) and career_recommendations else career_recommendations}"
    }
    
    # Create industry stability output
    industry_stability = {
        "status": stability_status,
        "trend": f"Current growth rate: {growth_rate}",
        "investment": f"Investment trend: {investment_trend}",
        "disruption_risk": f"Technological disruption risk: {disruption_risk}",
        "future_outlook": f"Projected {industry} industry change based on {source}"
    }
    
    return {
        "career_growth": career_growth,
        "industry_stability": industry_stability
    }

def get_fallback_industry_data(risk_level, industry="General"):
    """Provide more accurate industry-specific fallback data"""
    # Industry-specific growth data from BLS or other reliable sources
    industry_data = {
        "Technology": {
            "growth": "11.5%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Computer and Information Technology",
            "skills": ["Cloud computing", "Cybersecurity", "Software development"],
            "outlook": "Much faster than average growth",
            "recommendation": "Develop specialized skills in AI, machine learning, or cybersecurity"
        },
        "Healthcare": {
            "growth": "13%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Healthcare Occupations",
            "skills": ["Electronic health records", "Patient care", "Medical technology"],
            "outlook": "Much faster than average growth due to aging population",
            "recommendation": "Pursue certifications in specialized areas of patient care or healthcare technology"
        },
        "Finance": {
            "growth": "7%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Business and Financial Occupations",
            "skills": ["Financial analysis", "Risk management", "Regulatory compliance"],
            "outlook": "Faster than average growth",
            "recommendation": "Build expertise in fintech and regulatory frameworks"
        },
        "Education": {
            "growth": "4%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Education, Training, and Library Occupations",
            "skills": ["Online teaching", "Curriculum development", "Educational technology"],
            "outlook": "Average growth",
            "recommendation": "Develop skills in digital learning technologies and personalized education"
        },
        "Manufacturing": {
            "growth": "2%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Production Occupations",
            "skills": ["Automation systems", "Quality control", "Supply chain management"],
            "outlook": "Slower than average growth",
            "recommendation": "Focus on advanced manufacturing technologies and automation"
        },
        "Retail": {
            "growth": "3%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Sales Occupations",
            "skills": ["Customer service", "E-commerce", "Digital marketing"],
            "outlook": "Slower than average growth",
            "recommendation": "Develop omnichannel retail skills and data analytics capabilities"
        },
        "AI and Data": {
            "growth": "23%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Data Scientists and Mathematical Science Occupations",
            "skills": ["Machine learning", "Data analysis", "Python programming"],
            "outlook": "Much faster than average growth",
            "recommendation": "Build expertise in specialized AI domains like NLP or computer vision"
        },
        "Construction": {
            "growth": "4%", 
            "source": "Bureau of Labor Statistics, 2022-2032 projections for Construction and Extraction Occupations",
            "skills": ["Project management", "Blueprint reading", "Construction technology"],
            "outlook": "Average growth",
            "recommendation": "Develop skills in sustainable construction and building information modeling"
        },
        "General": {
            "growth": "5%", 
            "source": "Bureau of Labor Statistics, 2022-2032 average job growth projection across all occupations",
            "skills": ["Digital literacy", "Communication", "Problem solving"],
            "outlook": "Average growth across the economy",
            "recommendation": "Develop adaptable skills and continuous learning habits"
        }
    }
    
    # Get the specific industry data or fall back to general data
    industry_info = industry_data.get(industry, industry_data["General"])
    
    # Determine growth level
    growth_str = industry_info["growth"].replace("%", "")
    try:
        growth_pct = float(growth_str)
        if growth_pct > 10:
            growth_level = "High"
        elif growth_pct > 5:
            growth_level = "Moderate"
        else:
            growth_level = "Low"
    except:
        growth_level = "Moderate"
    
    # Adjust stability status based on risk level
    if risk_level > 0.6:
        stability_status = "Declining"
        disruption_risk = "High"
    elif risk_level > 0.3:
        stability_status = "Stable with changes"
        disruption_risk = "Medium"
    else:
        stability_status = "Growing"
        disruption_risk = "Low"
    
    # Create career growth output with real data
    career_growth = {
        "level": growth_level,
        "description": f"Based on {industry_info['source']}",
        "outlook": f"Projected growth: {industry_info['growth']} ({industry_info['outlook']})",
        "skill_demand": f"In-demand skills: {', '.join(industry_info['skills'])}",
        "recommendations": industry_info["recommendation"]
    }
    
    # Create industry stability output with real data
    industry_stability = {
        "status": stability_status,
        "trend": f"Current growth rate: {industry_info['growth']}",
        "investment": f"Investment trend: {stability_status}",
        "disruption_risk": f"Technological disruption risk: {disruption_risk}",
        "future_outlook": f"Based on {industry_info['source']}"
    }
    
    return {
        "career_growth": career_growth,
        "industry_stability": industry_stability
    }

def get_company_information(url):
    """Get information about the company from the job posting URL using Perplexity API"""
    if not PERPLEXITY_API_KEY:
        print("Warning: Perplexity API key not found. Using fallback company data.")
        return {
            "company_name": "Unknown",
            "industry": "Technology",
            "description": "Technology company",
            "growth_info": "Recent industry growth trends indicate continued expansion",
            "skills_in_demand": "Technical skills, problem-solving, communication"
        }
    
    print(f"Fetching company information from URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a targeted query to extract company information
    query = (
        f"Visit this job posting URL: {url} "
        f"Extract the following information: "
        f"1. Company name "
        f"2. Industry sector "
        f"3. Brief company description "
        f"4. Recent company growth or industry trends "
        f"5. Top skills in demand for this role "
        f"Format your response as plain text with clear section headers."
    )
    
    # Perplexity API payload
    payload = {
        "model": "sonar",  # Using sonar model for web browsing
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, 
                               json=payload,
                               timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract structured information from the response
            company_data = {}
            
            # Extract company name
            company_match = re.search(r'Company name:?\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            company_data["company_name"] = company_match.group(1).strip() if company_match else "Unknown"
            
            # Extract industry
            industry_match = re.search(r'Industry sector:?\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            company_data["industry"] = industry_match.group(1).strip() if industry_match else "Technology"
            
            # Extract company description
            desc_match = re.search(r'company description:?\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            company_data["description"] = desc_match.group(1).strip() if desc_match else "Technology company"
            
            # Extract growth info
            growth_match = re.search(r'(?:growth|trends):?\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            company_data["growth_info"] = growth_match.group(1).strip() if growth_match else "Recent industry growth trends indicate continued expansion"
            
            # Extract skills
            skills_match = re.search(r'skills:?\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            company_data["skills_in_demand"] = skills_match.group(1).strip() if skills_match else "Technical skills, problem-solving, communication"
            
            print("Successfully extracted company information")
            return company_data
            
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return {
                "company_name": "Unknown",
                "industry": "Technology",
                "description": "Technology company",
                "growth_info": "Recent industry growth trends indicate continued expansion",
                "skills_in_demand": "Technical skills, problem-solving, communication"
            }
            
    except Exception as e:
        print(f"Error fetching company information: {e}")
        return {
            "company_name": "Unknown",
            "industry": "Technology",
            "description": "Technology company",
            "growth_info": "Recent industry growth trends indicate continued expansion",
            "skills_in_demand": "Technical skills, problem-solving, communication"
        }

def create_career_growth_from_company_info(company_info, industry):
    """Create truthful career growth information from company information"""
    if not company_info:
        return {
            "level": "Unknown",
            "description": "Could not retrieve company information",
            "outlook": "Industry outlook data unavailable",
            "skill_demand": "Skill demand data unavailable",
            "recommendations": "Career recommendation data unavailable"
        }
    
    # Company name and industry for accurate reporting
    company_name = company_info.get("company_name", "Unknown")
    company_industry = company_info.get("industry", industry)
    
    # Use actual skills from the job posting
    skills = company_info.get("skills_in_demand", "").strip()
    if not skills or skills == "Technical skills, problem-solving, communication":
        skills = "Skills data unavailable"
    
    # Get real company growth information
    growth_info = company_info.get("growth_info", "").strip()
    
    # Determine growth level based on actual keywords in company growth info
    growth_level = "Moderate"  # Default
    outlook = "Standard industry growth expected"
    
    # Only assign "High" growth if explicitly stated in the job posting
    if growth_info and any(term in growth_info.lower() for term in [
        "rapid growth", "fast growing", "explosive growth", "significant expansion", 
        "high growth", "substantial growth", "growing quickly", "expansion"
    ]):
        growth_level = "High"
        outlook = f"Company reports strong growth: {growth_info}"
    # Only assign "Low" growth if explicitly stated
    elif growth_info and any(term in growth_info.lower() for term in [
        "decline", "shrinking", "downsizing", "layoffs", "challenges", 
        "restructuring", "difficult market"
    ]):
        growth_level = "Low"
        outlook = f"Company facing challenges: {growth_info}"
    # Otherwise use a balanced, factual statement
    elif growth_info:
        outlook = f"Company reports: {growth_info}"
    else:
        # Get general industry data if no company-specific info
        industry_growth = {
            "Technology": "11.5%", "Healthcare": "13%", "Finance": "7%",
            "Education": "4%", "Manufacturing": "2%", "Retail": "3%",
            "AI and Data": "23%", "Construction": "4%"
        }.get(industry, "5%")
        
        outlook = f"General {industry} industry growth projection: {industry_growth} (BLS data)"
    
    # Create a factual recommendation based on the actual company's description
    company_desc = company_info.get("description", "").strip()
    if company_desc and company_desc != "Technology company":
        recommendation = f"Consider how your skills align with {company_name}'s focus as a {company_desc}"
    else:
        recommendation = f"Research {company_name}'s specific growth areas and skill requirements"
    
    return {
        "level": growth_level,
        "description": f"Based on information from {company_name}'s job posting",
        "outlook": outlook,
        "skill_demand": f"Required skills: {skills}",
        "recommendations": recommendation
    }

def get_career_insights_from_perplexity(company_info, industry):
    """Use Perplexity API to analyze career growth potential based on company data"""
    if not PERPLEXITY_API_KEY or not company_info:
        print("Warning: Cannot generate career insights (missing API key or company data)")
        return create_career_growth_from_company_info(company_info, industry)
    
    company_name = company_info.get("company_name", "Unknown")
    company_description = company_info.get("description", "")
    skills = company_info.get("skills_in_demand", "")
    
    print(f"Fetching career growth insights for {company_name} in {industry} industry")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a targeted query for career growth analysis
    query = (
        f"Analyze career growth potential for jobs at {company_name}, a {company_description} in the {industry} industry. "
        f"Required skills include: {skills}. "
        f"Please provide: "
        f"1. Growth potential (High/Moderate/Low) based on factual information with reasoning "
        f"2. Industry outlook with specific growth percentage from Bureau of Labor Statistics or other reliable sources "
        f"3. Most valuable skills for long-term success in this role/company "
        f"4. One specific, actionable career recommendation for someone in this role "
        f"Provide only factual information from reliable sources. Include source citations. "
        f"Format as plain text with clearly labeled sections."
    )
    
    # Perplexity API payload
    payload = {
        "model": "sonar",  # Using sonar model for web search capabilities
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.1,  # Low temperature for factual responses
        "max_tokens": 1024
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                                headers=headers, 
                                json=payload,
                                timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse the response to extract structured information
            career_data = {}
            
            # Extract growth potential
            growth_match = re.search(r'Growth potential:?\s*(High|Moderate|Low)[^\n]*', content, re.IGNORECASE)
            level = "Moderate"  # Default
            if growth_match:
                level_text = growth_match.group(1).strip()
                if "high" in level_text.lower():
                    level = "High"
                elif "low" in level_text.lower():
                    level = "Low"
                else:
                    level = "Moderate"
            
            # Extract description/reasoning
            desc_match = re.search(r'Growth potential:.*?reasoning:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', content, re.IGNORECASE | re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else f"Based on analysis of {company_name} in the {industry} industry"
            
            # Extract outlook
            outlook_match = re.search(r'Industry outlook:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', content, re.IGNORECASE | re.DOTALL)
            outlook = outlook_match.group(1).strip() if outlook_match else f"General {industry} industry growth"
            
            # Extract skills
            skills_match = re.search(r'(?:valuable|important|in-demand) skills:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', content, re.IGNORECASE | re.DOTALL)
            skill_demand = skills_match.group(1).strip() if skills_match else f"Required skills: {skills}"
            
            # Extract recommendation
            rec_match = re.search(r'(?:recommendation|action|advice):?\s*(.*?)(?:\n\n|\n[A-Z]|$)', content, re.IGNORECASE | re.DOTALL)
            recommendation = rec_match.group(1).strip() if rec_match else f"Develop specialized skills relevant to {company_name}"
            
            # Look for source citations
            sources = re.findall(r'(?:source|according to|based on)[:\s]+((?:Bureau of Labor Statistics|BLS|U\.S\. Department|LinkedIn|Indeed|Glassdoor|[A-Za-z\s]+Report)[^\.]+)', content, re.IGNORECASE)
            
            # Add source attribution if found
            source_text = ""
            if sources:
                source_text = f" (Source: {sources[0].strip()})"
                
            return {
                "level": level,
                "description": f"{description}{source_text}",
                "outlook": outlook,
                "skill_demand": skill_demand,
                "recommendations": recommendation
            }
        else:
            print(f"Perplexity API error: {response.status_code} - {response.text}")
            # Fall back to simpler method
            return create_career_growth_from_company_info(company_info, industry)
            
    except Exception as e:
        print(f"Error getting career insights: {e}")
        return create_career_growth_from_company_info(company_info, industry)

if __name__ == "__main__":
    # Path to your data
    csv_path = ECONOMIC_DATA_FILE
    
    # Get job description
    print("Enter job description (press Enter twice when done):")
    lines = []
    line = input()
    while line:
        lines.append(line)
        line = input()
    
    job_description = "\n".join(lines)
    
    # Analyze
    results = analyze_job_automation(job_description, csv_path)
    
    # Print results
    print("\nResults:")
    print(f"Detected Industry: {results.get('Industry', 'Unknown')}")
    print(f"Automation Risk: {results.get('Overall_Automation_Risk', 0):.4f}")
    
    # Enhanced career growth output
    career = results.get('Career_Growth_Potential', {})
    print(f"\nCareer Growth Potential: {career.get('level', 'Unknown')}")
    print(f"  • {career.get('description', '')}")
    print(f"  • Outlook: {career.get('outlook', '')}")
    print(f"  • Skill Demand: {career.get('skill_demand', '')}")
    print(f"  • Recommendation: {career.get('recommendations', '')}")
    
    # Enhanced industry stability output
    industry = results.get('Industry_Stability', {})
    print(f"\nIndustry Stability: {industry.get('status', 'Unknown')}")
    print(f"  • Trend: {industry.get('trend', '')}")
    print(f"  • Investment: {industry.get('investment', '')}")
    print(f"  • Disruption Risk: {industry.get('disruption_risk', '')}")
    print(f"  • Future Outlook: {industry.get('future_outlook', '')}")
    
    # Show top matches
    if 'Matched_Tasks' in results:
        print("\nTop matching tasks:")
        for i, task in enumerate(results['Matched_Tasks']):
            print(f"{i+1}. '{task['job_task'][:50]}...'")
            print(f"   Matched to: '{task['matched_task'][:50]}...'")
            print(f"   Automation: {task['automation_pct']*100:.2f}%")
            print(f"   Match confidence: {task['match_score']:.2f}")
