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

def analyze_job_automation(job_posting_url, csv_path=None, similarity_threshold=0.3):
    """Analyze job automation risk based on Claude usage patterns from a job posting URL"""
    try:
        # Use existing job_fetcher functionality
        job_description = fetch_job_from_url(job_posting_url)
        
        if not job_description:
            return {"error": f"Failed to extract job description from URL: {job_posting_url}"}
            
        # Load task data
        task_data = load_task_data(csv_path)
        if task_data.empty:
            return {"error": "Failed to load task data"}

        # Extract tasks from job description
        job_tasks = extract_tasks(job_description)
        if not job_tasks:
            job_tasks = [job_description]  # Use full description if no tasks extracted
            
        # Detect industry
        industry = detect_industry(job_description)
        
        # Find matching tasks and their Claude usage
        matched_tasks = find_matching_tasks(job_tasks, task_data, similarity_threshold)
        
        # Identify high-usage tasks (most likely to be automated)
        high_usage_tasks = []
        medium_usage_tasks = []
        low_usage_tasks = []
        
        # Define usage thresholds
        HIGH_USAGE_THRESHOLD = 0.01681  # Top 25%
        LOW_USAGE_THRESHOLD = 0.00326   # Bottom 25%
        
        # Categorize tasks by usage
        for task in matched_tasks:
            usage = task.get('claude_usage_pct', 0.01)
            if usage >= HIGH_USAGE_THRESHOLD:
                high_usage_tasks.append(task)
            elif usage <= LOW_USAGE_THRESHOLD:
                low_usage_tasks.append(task)
            else:
                medium_usage_tasks.append(task)
        
        # Calculate overall risk based on proportion of high-usage tasks
        if matched_tasks:
            high_usage_proportion = len(high_usage_tasks) / len(matched_tasks)
            # Scale to risk score (0.1 to 0.9)
            base_risk = 0.1 + (high_usage_proportion * 0.8)
        else:
            base_risk = 0.5  # Default risk
        
        # Get industry growth rate
        growth_rate = parse_growth_rate(get_industry_growth_rate(industry))
        
        # Adjust risk based on industry growth (higher growth = lower risk)
        growth_adjustment = min(0.2, max(-0.2, (growth_rate - 4.0) / 20.0))
        adjusted_risk = min(0.9, max(0.1, base_risk - growth_adjustment))
        
        # Generate reasoning
        reasoning = generate_simple_reasoning(
            high_usage_tasks, 
            medium_usage_tasks,
            low_usage_tasks,
            growth_rate,
            industry
        )
        
        # Extract company info from URL
        from urllib.parse import urlparse
        domain = urlparse(job_posting_url).netloc
        company_name = domain.split('.')[0].replace('-', ' ').title()
        if company_name.lower() in ['www', 'jobs', 'careers']:
            company_name = domain.split('.')[1].replace('-', ' ').title()
            
        company_info = {
            'company_name': company_name,
            'industry': industry,
            'skills_in_demand': extract_skills(job_description),
            'job_url': job_posting_url
        }
        
        # Get career growth potential
        career_growth = analyze_company_growth(company_info, industry)
        
        # Compile results
        results = {
            'Industry': industry,
            'Overall_Automation_Risk': adjusted_risk,
            'Risk_Reasoning': reasoning,
            'Matched_Tasks': matched_tasks,
            'Company_Info': company_info,
            'Career_Growth_Potential': career_growth
        }
        
        return results
        
    except Exception as e:
        print(f"Error in job analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Analysis error: {str(e)}"}

def find_matching_tasks(job_tasks, task_data, similarity_threshold=0.3):
    """Find matching tasks and their Claude usage"""
    matched_tasks = []
    
    # Ensure 'claude_usage_pct' column exists
    if 'claude_usage_pct' not in task_data.columns and 'pct' in task_data.columns:
        task_data['claude_usage_pct'] = task_data['pct']
    elif 'claude_usage_pct' not in task_data.columns:
        print("Warning: No Claude usage data found. Using default values.")
        task_data['claude_usage_pct'] = 0.01  # Default value
    
    for job_task in job_tasks:
        best_match = None
        best_score = 0
        
        for _, row in task_data.iterrows():
            task_name = row['task_name']
            claude_usage = row['claude_usage_pct']
            
            similarity = calculate_similarity(job_task, task_name)
            
            if similarity > similarity_threshold and similarity > best_score:
                best_score = similarity
                best_match = {
                    'job_task': job_task,
                    'matched_task': task_name,
                    'match_score': similarity,
                    'claude_usage_pct': claude_usage
                }
        
        if best_match:
            matched_tasks.append(best_match)
    
    return matched_tasks

def generate_simple_reasoning(high_tasks, medium_tasks, low_tasks, growth_rate, industry):
    """Generate simple reasoning based on task usage patterns"""
    
    # Count tasks in each category
    high_count = len(high_tasks)
    medium_count = len(medium_tasks)
    low_count = len(low_tasks)
    total_count = high_count + medium_count + low_count
    
    if total_count == 0:
        return "No tasks could be analyzed for automation potential."
    
    # Calculate percentages
    high_pct = (high_count / total_count) * 100 if total_count > 0 else 0
    
    # Start building reasoning
    reasoning = []
    
    # Add high usage task information
    if high_count > 0:
        reasoning.append(f"{high_count} tasks ({high_pct:.1f}%) have HIGH usage in Claude interactions")
        reasoning.append("These tasks are most likely to be automated:")
        for task in high_tasks[:3]:  # Show top 3
            reasoning.append(f"• '{task['job_task'][:50]}...'")
    
    # Add industry growth information
    if growth_rate >= 8.0:
        reasoning.append(f"Industry growth is strong ({growth_rate}%), which may offset automation risk")
    elif growth_rate <= 2.0:
        reasoning.append(f"Industry growth is slow ({growth_rate}%), which may increase automation risk")
    else:
        reasoning.append(f"Industry growth is moderate ({growth_rate}%)")
    
    # Add summary
    if high_count > medium_count + low_count:
        reasoning.append("Overall: Most job tasks have high automation potential")
    elif high_count == 0:
        reasoning.append("Overall: Job tasks have low automation potential")
    else:
        reasoning.append("Overall: Job has mixed automation potential")
    
    return "\n".join(reasoning)

def load_task_data(csv_path=None):
    """Load task data from CSV with proper error handling"""
    try:
        if csv_path is None:
            csv_path = ONET_TASK_MAPPINGS_FILE
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        task_data = pd.read_csv(csv_path)
        return task_data
    except Exception as e:
        print(f"Error loading task data: {e}")
        return pd.DataFrame()

def create_error_response(error_msg):
    """Create standardized error messages"""
    return {
        "Error": error_msg
    }

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def parse_growth_rate(growth_info):
    """Parse growth rate from various formats"""
    if isinstance(growth_info, str):
        if "%" in growth_info:
            try:
                return float(growth_info.strip("%"))
            except ValueError:
                pass
        elif "percent" in growth_info.lower():
            return 0.0  # Assuming 0% growth
    elif isinstance(growth_info, (int, float)):
        return growth_info
    return 4.0  # Default to 4% growth if no valid format found

def get_industry_growth_rate(industry):
    """Get industry growth rate based on industry name"""
    # This function should be implemented to return the actual growth rate based on the industry
    # For now, we'll use a placeholder
    return "4%"

def scrape_job_posting(url):
    """Get information about the company from the job posting URL using Perplexity API"""
    
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
        f"5. Top skills in demand for this role. Make sure these are actionable skills rather than qualifications like a degree "
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
            
    except Exception as e:
        print(f"Error fetching company information: {e}")
        return {
            "company_name": "Unknown",
            "industry": "Technology",
            "description": "Technology company",
            "growth_info": "Unknown",
            "skills_in_demand": "Unknown"
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

def get_career_insights_(company_info, industry):
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
    # Load BLS employment projections data
    try:
        bls_projections = pd.read_csv('insights/data/bls_employment_projection_2023_33_annualized_rounded.csv')
        
        # Map industry to BLS occupational groups
        industry_to_bls = {
            'Technology': ['Computer and mathematical', 'Information technology'],
            'Healthcare': ['Healthcare practitioners and technical', 'Healthcare support'],
            'Finance': ['Business and financial operations'],
            'Education': ['Educational instruction and library'],
            'Manufacturing': ['Production'],
            'Retail': ['Sales and related'],
            'Construction': ['Construction and extraction'],
            'Engineering': ['Architecture and engineering'],
            'Legal': ['Legal'],
            'Science': ['Life, physical, and social science']
        }
        
        # Find matching BLS occupational groups
        matching_groups = []
        for key, groups in industry_to_bls.items():
            if industry.lower() in key.lower():
                matching_groups.extend(groups)
        
        if matching_groups:
            # Get growth projections for matching groups
            growth_data = bls_projections[bls_projections['Occupational Group'].str.lower().isin(
                [group.lower() for group in matching_groups]
            )]
            
            if not growth_data.empty:
                # Calculate average projected growth
                avg_growth = growth_data['Projected Employment Change (%)'].mean()
                print(f"Found BLS growth projection for {industry}: {avg_growth:.1f}%")
                company_info['growth'] = f"{avg_growth:.1f}%"
                
                # Add growth level based on BLS data
                if avg_growth > 10:
                    company_info['growth_level'] = 'High'
                elif avg_growth > 5:
                    company_info['growth_level'] = 'Moderate'
                else:
                    company_info['growth_level'] = 'Low'
            
    except Exception as e:
        print(f"Error loading BLS projections: {e}")
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

def format_results_for_discord(results):
    """
    Format analysis results for Discord display with improved output
    """
    output = []
    
    # Header with automation risk
    risk = results.get('Overall_Automation_Risk', 0)
    risk_text = ("LOW" if risk < 0.3 else 
                "MODERATE" if risk < 0.6 else 
                "HIGH")
    risk_emoji = "🟢" if risk < 0.3 else "🟠" if risk < 0.6 else "🔴"
    
    output.append(f"{risk_emoji} **Automation Risk: {risk_text} ({risk:.2f})**")
    output.append(results.get('Risk_Reasoning', ''))
    
    # Career growth potential section with industry-specific data
    career = results.get('Career_Growth_Potential', {})
    output.append("\n📈 **Career Growth Potential**")
    output.append(f"**Level**: {career.get('level', 'Unknown')}")
    output.append(f"• {career.get('description', '')}")
    
    # Add specific growth rate data if available
    if 'growth_rate' in career:
        output.append(f"• **Industry Growth Rate**: {career.get('growth_rate', 'Unknown')}")
    
    if 'employment_info' in career and career['employment_info']:
        output.append(f"• {career.get('employment_info', '')}")
        
    output.append(f"• Outlook: {career.get('outlook', '')}")
    output.append(f"• Key skills: {career.get('skill_demand', '')}")
    
    # Make sure recommendations are fully displayed
    if 'recommendations' in career:
        # Split recommendations by comma to format them nicely
        recs = career.get('recommendations', '').split(',')
        output.append("• **Recommendations**:")
        for rec in recs:
            output.append(f"  ↳ {rec.strip()}")
    
    # Industry stability section
    industry = results.get('Industry_Stability', {})
    output.append(f"\n🏢 **Industry: {results.get('Industry', 'Unknown')}**")
    output.append(f"• Status: {industry.get('status', 'Unknown')}")
    output.append(f"• Disruption Risk: {industry.get('disruption_risk', '')}")
    output.append(f"• Future Outlook: {industry.get('future_outlook', '')}")
    
    # Add matched tasks with better formatting
    if 'Matched_Tasks' in results and results['Matched_Tasks']:
        output.append("\n**Job Tasks Analysis:**")
        for i, task in enumerate(results['Matched_Tasks'][:5]):  # Show top 5 tasks
            emoji = "🟢" if task['automation_pct'] < 0.3 else "🟠" if task['automation_pct'] < 0.6 else "🔴"
            # Truncate long task descriptions but keep more content
            task_desc = task['job_task']
            if len(task_desc) > 100:
                task_desc = task_desc[:97] + "..."
                
            output.append(f"{i+1}. {emoji} **Task**: {task_desc}")
            output.append(f"   → Automation: {task['automation_pct']*100:.1f}% (Confidence: {task['match_score']:.2f})")
    
    return "\n".join(output)

def analyze_company_growth(company_info, industry):
    """Analyze company growth using BLS data and company information"""
    try:
        company_name = company_info.get("company_name", "Unknown")
        growth_info = company_info.get("growth_info", "").strip()
        skills = company_info.get("skills_in_demand", "").strip()
        
        # Load BLS projections
        df = pd.read_csv('insights/data/bls_employment_projection_2023_33_annualized_rounded.csv')
        
        # Get industry growth rate from BLS data
        industry_row = df[df['Occupational Group'].str.lower().str.contains(industry.lower(), na=False)]
        if not industry_row.empty:
            growth_rate = industry_row['Projected Employment Change (%)'].iloc[0]
            growth_level = "High" if growth_rate > 8.0 else "Low" if growth_rate < 2.0 else "Moderate"
            industry_outlook = f"Industry projected growth: {growth_rate}% (BLS data)"
        else:
            growth_rate = 4.0  # Overall job market growth
            growth_level = "Moderate"
            industry_outlook = f"Overall job market growth: {growth_rate}% (BLS data)"
        
        # Combine company and industry information
        outlook = (f"Company information: {growth_info}\n{industry_outlook}" if growth_info 
                  else industry_outlook)
        
        return {
            "level": growth_level,
            "description": f"Based on BLS projections and {company_name} data",
            "outlook": outlook,
            "skill_demand": f"Required skills: {skills}",
            "recommendations": f"Research growth opportunities at {company_name} and industry trends"
        }
        
    except Exception as e:
        print(f"Error analyzing growth: {e}")
        return {
            "level": "Unknown",
            "description": "Unable to analyze growth data",
            "outlook": "Data unavailable",
            "skill_demand": f"Required skills: {skills}",
            "recommendations": "Research company and industry trends"
        }

def get_generic_career_growth_for_industry(industry, risk_level):
    """Generate industry-appropriate career guidance using BLS data without hard-coding"""
    try:
        # Load BLS employment projections data
        bls_file = 'insights/data/bls_employment_may_2023.csv'
        if not os.path.exists(bls_file):
            print(f"Warning: BLS data file not found: {bls_file}")
            bls_data = pd.DataFrame()
        else:
            bls_data = pd.read_csv(bls_file)
            print(f"Loaded BLS data with {len(bls_data)} records")
        
        # Find matching industry data
        industry_info = None
        growth_rate = None
        employment_info = None
        
        if not bls_data.empty:
            # Try to find the industry in the data
            # Assuming the BLS data has columns like 'Title' and 'Growth'
            possible_columns = ['Title', 'Occupation', 'Occupational Group', 'Industry']
            title_col = next((col for col in possible_columns if col in bls_data.columns), None)
            
            if title_col:
                # Look for the industry in the data
                pattern = '|'.join(industry.split())
                matches = bls_data[bls_data[title_col].str.contains(pattern, case=False, na=False)]
                
                if not matches.empty:
                    # Get the first matching row
                    industry_info = matches.iloc[0]
                    
                    # Extract growth rate if available
                    growth_cols = [col for col in bls_data.columns if 'growth' in col.lower() or 'change' in col.lower()]
                    if growth_cols:
                        growth_rate = industry_info[growth_cols[0]]
                        
                    # Extract employment data if available
                    emp_cols = [col for col in bls_data.columns if 'employ' in col.lower()]
                    if emp_cols and len(emp_cols) >= 2:
                        current = industry_info[emp_cols[0]]
                        projected = industry_info[emp_cols[1]]
                        employment_info = f"Employment data: {current} (current) to {projected} (projected)"
        
        # Determine growth level based on BLS data (if available) and risk level
        if growth_rate is not None:
            try:
                growth_rate = float(str(growth_rate).replace('%', ''))
                if growth_rate > 8.0:
                    level = "High" if risk_level < 0.6 else "Moderate"
                    outlook = f"Projected growth of {growth_rate}% based on BLS data"
                elif growth_rate < 2.0:
                    level = "Low" if risk_level > 0.4 else "Moderate"
                    outlook = f"Limited projected growth of {growth_rate}% based on BLS data"
                else:
                    level = "Moderate"
                    outlook = f"Average projected growth of {growth_rate}% based on BLS data"
            except (ValueError, TypeError):
                growth_rate = None
                
        # If no specific BLS data found, use risk level to determine growth outlook
        if growth_rate is None:
            if risk_level > 0.6:
                level = "Uncertain"
                outlook = "High automation risk may impact job stability"
                growth_rate = "Data unavailable"
            elif risk_level > 0.3:
                level = "Moderate"
                outlook = "Moderate automation risk suggests evolving job functions"
                growth_rate = "Data unavailable" 
            else:
                level = "Favorable"
                outlook = "Low automation risk suggests stable job functions"
                growth_rate = "Data unavailable"
                
        # Get skill recommendations based on industry and risk level
        skills = get_skills_for_industry(industry, risk_level, bls_data)
        recommendations = get_recommendations(industry, risk_level, growth_rate)
        
        return {
            "level": level,
            "description": f"Based on available data for {industry}",
            "outlook": outlook,
            "growth_rate": str(growth_rate) + "%" if isinstance(growth_rate, (int, float)) else growth_rate,
            "employment_info": employment_info,
            "skill_demand": skills,
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"Error in career growth analysis: {e}")
        return {
            "level": "Unknown",
            "description": "Unable to determine based on available data",
            "outlook": "Data unavailable",
            "growth_rate": "Data unavailable",
            "skill_demand": "Focus on adaptable skills like critical thinking and problem-solving",
            "recommendations": "Develop transferable skills and stay current with technology trends"
        }

def get_skills_for_industry(industry, risk_level, bls_data=None):
    """Get relevant skills for an industry based on data, not hard-coding"""
    try:
        # Try to load skills data if available
        skills_file = 'insights/data/industry_skills.csv'
        if os.path.exists(skills_file):
            skills_data = pd.read_csv(skills_file)
            
            # Find matching industry
            matches = skills_data[skills_data['industry'].str.contains(industry, case=False, na=False)]
            if not matches.empty:
                # Extract skills based on automation risk
                if risk_level > 0.6:
                    skill_col = 'high_automation_skills'
                elif risk_level > 0.3:
                    skill_col = 'medium_automation_skills'
                else:
                    skill_col = 'low_automation_skills'
                    
                if skill_col in matches.columns:
                    return matches[skill_col].iloc[0]
        
        # If no specific skills found, provide general skills based on risk level
        if risk_level > 0.6:
            return "Focus on creative problem-solving, strategic thinking, and complex communication skills"
        elif risk_level > 0.3:
            return "Develop technical skills balanced with interpersonal capabilities and domain expertise"
        else:
            return "Strengthen specialized knowledge, collaborative skills, and adaptability"
            
    except Exception as e:
        print(f"Error getting skills: {e}")
        return "Skills in critical thinking, adaptability, and technical literacy remain valuable across industries"

def get_recommendations(industry, risk_level, growth_rate):
    """Generate recommendations based on data files without hard-coding"""
    try:
        # Try to load recommendation data if available
        rec_file = 'insights/data/career_recommendations.csv'
        
        # Create file if it doesn't exist (first run)
        if not os.path.exists(rec_file):
            create_default_recommendations_file(rec_file)
            
        if os.path.exists(rec_file):
            rec_data = pd.read_csv(rec_file)
            
            # Find matching industry and risk level
            risk_category = 'high_risk' if risk_level > 0.6 else 'medium_risk' if risk_level > 0.3 else 'low_risk'
            
            # Try exact match first
            matches = rec_data[(rec_data['industry'] == industry) & 
                             (rec_data['risk_category'] == risk_category)]
                             
            # If no exact match, try partial match
            if matches.empty:
                matches = rec_data[(rec_data['industry'].str.contains(industry, case=False, na=False)) & 
                                 (rec_data['risk_category'] == risk_category)]
            
            # If still no match, try just the risk level
            if matches.empty:
                matches = rec_data[(rec_data['industry'] == 'General') & 
                                 (rec_data['risk_category'] == risk_category)]
            
            if not matches.empty:
                return matches['recommendations'].iloc[0]
        
        # If no specific recommendations found, generate based on risk and industry
        if risk_level > 0.6:
            return f"1) Focus on skills complementary to automation, 2) Consider roles requiring human judgment, 3) Stay current with {industry} technological trends"
        elif risk_level > 0.3:
            return f"1) Develop both technical and domain expertise, 2) Build communication and leadership skills, 3) Learn to effectively use automation tools in {industry}"
        else:
            return f"1) Deepen specialized knowledge in {industry}, 2) Develop management and strategic capabilities, 3) Focus on areas requiring human expertise and judgment"
    
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return f"Focus on continuous learning and staying adaptable to changes in {industry}"

def create_default_recommendations_file(filename):
    """Create a default recommendations file with some basic data"""
    data = [
        {'industry': 'Technology', 'risk_category': 'high_risk', 
         'recommendations': '1) Develop skills in AI ethics and governance, 2) Focus on AI-human collaboration roles, 3) Build expertise in areas requiring creativity and innovation'},
        {'industry': 'Technology', 'risk_category': 'medium_risk',
         'recommendations': '1) Balance technical skills with domain expertise, 2) Develop communication and leadership capabilities, 3) Learn to effectively use AI tools'},
        {'industry': 'Technology', 'risk_category': 'low_risk',
         'recommendations': '1) Deepen specialized technical knowledge, 2) Build cross-functional expertise, 3) Focus on innovation and system design'},
        
        {'industry': 'General', 'risk_category': 'high_risk',
         'recommendations': '1) Focus on human-centered skills (empathy, creativity, judgment), 2) Consider retraining for growing fields, 3) Learn how to collaborate with automation systems'},
        {'industry': 'General', 'risk_category': 'medium_risk',
         'recommendations': '1) Develop adaptable skill sets, 2) Balance technical and interpersonal capabilities, 3) Stay updated on industry technological trends'},
        {'industry': 'General', 'risk_category': 'low_risk',
         'recommendations': '1) Strengthen your specialized expertise, 2) Develop leadership capabilities, 3) Focus on innovation within your field'}
    ]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Created default recommendations file: {filename}")

def get_industry_stability(risk_level):
    """
    Generate industry stability assessment based on risk level without hard-coded data
    
    Args:
        risk_level: The calculated automation risk level
        
    Returns:
        Dictionary containing industry stability assessment
    """
    # Determine stability based on risk level
    if risk_level > 0.6:
        stability_status = "Changing rapidly"
        disruption_risk = "High"
        investment_trend = "Focused on automation technologies"
        future_outlook = "Significant workforce transformation expected"
    elif risk_level > 0.3:
        stability_status = "Evolving"
        disruption_risk = "Medium"
        investment_trend = "Balanced between technology and human capital"
        future_outlook = "Gradual evolution of job roles expected"
    else:
        stability_status = "Relatively stable"
        disruption_risk = "Lower"
        investment_trend = "Emphasis on human capital development"
        future_outlook = "Job roles likely to evolve but not be eliminated"
    
    return {
        "status": stability_status,
        "trend": "Economic conditions vary by region and specific sector",
        "investment": investment_trend,
        "disruption_risk": disruption_risk,
        "future_outlook": future_outlook
    }

def match_tasks_improved(job_tasks, task_data, similarity_threshold=0.05):
    """Match job tasks to dataset tasks with improved accuracy and better fallbacks"""
    return find_matching_tasks(job_tasks, task_data, similarity_threshold)

def extract_tasks_improved(job_description):
    """
    Extract tasks from job description using configurable patterns
    
    Args:
        job_description: Job description text
        
    Returns:
        List of task strings
    """
    # Load patterns from config file if it exists
    patterns_file = 'insights/data/task_patterns.json'
    if os.path.exists(patterns_file):
        with open(patterns_file, 'r') as f:
            config = json.load(f)
        patterns = config.get('patterns', [])
        filter_length = config.get('min_length', 15)
        excluded_patterns = config.get('excluded_patterns', [])
        action_verbs = config.get('action_verbs', [])
    else:
        # Default patterns if file doesn't exist
        patterns = [
            r'•\s*(.*?)(?=•|\n\n|\n[0-9•]|\Z)',  # Bullet points
            r'(?<!\S)(?<!\d)(?:\d+\.|\d+\))\s*(.*?)(?=\n\n|\n(?:\d+\.|\d+\))|\Z)',  # Numbered items
            r'(?<=\n)(?!-|-\s|•|\s*\d+\.|\s*\d+\))([A-Z][^.!?\n]*(?:[.!?]+|(?=\n)))'  # Sentences after newline
        ]
        filter_length = 15
        excluded_patterns = [r'^(?:location|salary|type|posted|apply|degree):']
        action_verbs = ['develop', 'create', 'manage', 'design', 'implement', 'build', 'analyze', 
                       'research', 'write', 'communicate', 'lead', 'collaborate', 'coordinate',
                       'maintain', 'ensure', 'provide', 'support', 'work', 'handle', 'prepare']
        
        # Create the config file for future use
        config = {
            'patterns': patterns,
            'min_length': filter_length,
            'excluded_patterns': excluded_patterns,
            'action_verbs': action_verbs
        }
        os.makedirs(os.path.dirname(patterns_file), exist_ok=True)
        with open(patterns_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    tasks = []
    
    # Try to find responsibilities section
    section_headers = r'\n\s*(?:Key )?(?:Responsibilities|Requirements|Qualifications|Skills|What You\'ll Do):\s*\n'
    sections = re.split(section_headers, job_description, flags=re.IGNORECASE)
    
    # Process each pattern on either the full text or responsibility sections if found
    text_to_process = sections if len(sections) > 1 else [job_description]
    
    for text in text_to_process:
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                task = match.group(1).strip()
                # Filter out short or non-task items
                if (len(task) > filter_length and
                    not any(re.match(ex_pattern, task.lower()) for ex_pattern in excluded_patterns) and
                    not task.startswith('http') and  # Not a URL
                    any(verb in task.lower() for verb in action_verbs)):
                    tasks.append(task)
    
    # If we still have no tasks, try to extract sentences
    if not tasks:
        sentences = re.findall(r'[^.!?\n]+[.!?]', job_description)
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(verb in sentence.lower() for verb in action_verbs[:7])):  # Use first 7 verbs for sentence matching
                tasks.append(sentence)
    
    return tasks

def detect_industry(job_description):
    """
    Industry detection using configuration file instead of hard-coded values
    
    Args:
        job_description: The job description text
        
    Returns:
        String with the detected industry
    """
    try:
        # Load industry keywords from config file
        industry_file = 'insights/data/industry_keywords.csv'
        if not os.path.exists(industry_file):
            # If file doesn't exist, create a minimal version for future use
            create_default_industry_file(industry_file)
            
        # Read the keywords from file
        industry_data = pd.read_csv(industry_file)
        
        # Convert to lowercase for case-insensitive matching
        text = job_description.lower()
        
        # Count keyword matches for each industry
        matches = {}
        for _, row in industry_data.iterrows():
            industry = row['industry']
            # Extract keywords (comma-separated in the CSV)
            keywords = [k.strip() for k in row['keywords'].split(',')]
            # Count matches
            count = sum(1 for keyword in keywords if keyword in text)
            if count > 0:
                matches[industry] = count
                
        # Check for company indicators
        company_file = 'insights/data/company_industries.csv'
        if os.path.exists(company_file):
            company_data = pd.read_csv(company_file)
            for _, row in company_data.iterrows():
                company = row['company'].lower()
                industry = row['industry']
                if company in text:
                    # If company name is found, heavily weight that industry
                    matches[industry] = matches.get(industry, 0) + 10
        
        # Special case handling from config
        special_case_file = 'insights/data/industry_special_cases.csv'
        if os.path.exists(special_case_file):
            special_cases = pd.read_csv(special_case_file)
            for _, row in special_cases.iterrows():
                pattern = row['pattern'].lower()
                result = row['industry']
                if pattern in text:
                    return result
                    
        # If we have matches, return the industry with the most matches
        if matches:
            return max(matches, key=matches.get)
            
        return "General"
    except Exception as e:
        print(f"Error in industry detection: {e}")
        return "General"

def create_default_industry_file(filename):
    """Create a default industry keywords file if none exists"""
    industries = {
        'Technology': 'software,tech,data,engineer,developer,coding,programming,IT,computer,algorithm,cloud,digital,cyber,web,app,platform,database,ai,artificial intelligence,machine learning,analytics',
        'Healthcare': 'health,medical,doctor,nurse,patient,clinical,pharma,biotech,hospital,care,therapy,treatment,diagnostic',
        'Finance': 'finance,bank,investment,financial,trading,asset,wealth,budget,accounting,audit,tax,compliance,risk',
        'Education': 'education,teach,school,student,learning,academic,course,curriculum,professor,faculty,university,college,training',
        'Consulting': 'consult,client,strategy,advisor,business solution,engagement,stakeholder,management consulting,professional services',
        'Manufacturing': 'manufacturing,product,assembly,production,factory,quality,supply chain,inventory,materials,industrial',
        'Retail': 'retail,store,shop,e-commerce,customer,merchandise,consumer,sales,inventory'
    }
    
    # Create dataframe and save to file
    data = []
    for industry, keywords in industries.items():
        data.append({'industry': industry, 'keywords': keywords})
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Created default industry keywords file: {filename}")

def extract_skills(job_description):
    """Extract skills from job description"""
    import re
    # Look for skills section
    skills_pattern = re.compile(r'(?:skills|requirements|qualifications)(?:[\s\:\-]+)([^\.]+)', re.IGNORECASE)
    skills_match = skills_pattern.search(job_description)
    
    if skills_match:
        return skills_match.group(1).strip()
    
    # Default skills
    return "Technical skills, problem-solving, communication"

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
    
    # Format and print results
    formatted_results = format_results_for_discord(results)
    print(formatted_results)
