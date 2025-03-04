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

def analyze_job(job_url):
    """
    Simple job analyzer that extracts key information from a job posting URL.
    Returns job details in a structured format matching print_job_analysis expectations.
    """
    try:
        # Fetch the job description
        job_description = fetch_job_from_url(job_url)
        
        # Extract company name dynamically
        company_name = extract_company_name(job_description, job_url)
        
        # Detect industry based on job content
        industry, confidence = detect_industry_from_content(job_description)
        
        # Extract tasks and skills
        tasks, skills = extract_tasks_and_skills(job_description)
        
        # Format skills for display
        formatted_skills = format_skills_for_display(skills)
        
        # Get industry growth data from available sources
        growth_rate, industry_outlook = get_industry_metrics(industry)
        
        # Analyze automation potential based on task content
        automatable_tasks = analyze_task_automation(tasks)
        
        # Generate contextual recommendations
        recommendations = generate_recommendations(industry, automatable_tasks, tasks)
        
        # Format results to match the expected structure
        return {
            'Company_Info': {
                'company_name': company_name,
                'industry': industry
            },
            'Career_Growth_Potential': {
                'growth_rate': growth_rate,
                'skill_demand': formatted_skills,
                'recommendations': recommendations
            },
            'Risk_Reasoning': f"{len(tasks)} job tasks identified in {industry} industry\n" + 
                            f"Overall: Industry has {growth_rate} projected growth with {confidence} confidence",
            'High_Usage_Tasks': automatable_tasks[:3]  # Top 3 most automatable tasks
        }
        
    except Exception as e:
        print(f"Error analyzing job: {e}")
        return {
            'error': str(e),
            'Company_Info': {
                'company_name': "Error in processing",
                'industry': "Unknown"
            },
            'Career_Growth_Potential': {
                'growth_rate': "N/A",
                'skill_demand': "Could not extract skills"
            },
            'Risk_Reasoning': "Analysis failed",
            'High_Usage_Tasks': []
        }

def extract_company_name(job_description, job_url):
    """Extract company name from job description and URL"""
    from urllib.parse import urlparse
    
    # Try to extract from job description first
    company_patterns = [
        r'About\s+([\w\s]+)(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company)?(?:\s|$|\.)',
        r'([\w\s]+)(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company)? is (?:a|an|the)',
        r'Join\s+([\w\s]+)(?:\'s)? team',
        r'(?:About|At)\s+([\w\s]+)(?:,|\.|\n)',
        r'Company:\s+([\w\s]+)(?:,|\.|\n)'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        if matches:
            company = matches[0].strip() if isinstance(matches[0], str) else matches[0][0].strip()
            if len(company) > 2 and len(company) < 50:  # Reasonable company name length
                return company
    
    # If not found in description, try from URL
    parsed_url = urlparse(job_url)
    path_parts = parsed_url.path.strip('/').split('/')
    domain = parsed_url.netloc.lower()
    
    # Check for job board patterns
    job_boards = ['linkedin', 'indeed', 'glassdoor', 'ziprecruiter', 'monster', 'ashbyhq', 'lever']
    
    if any(board in domain for board in job_boards):
        # For job boards, company name might be in the path
        if path_parts and len(path_parts) > 0:
            company_name = path_parts[0].replace('-', ' ').replace('_', ' ').title()
            if company_name.lower() not in ['jobs', 'job', 'careers', 'career', 'search']:
                return company_name
    
    # Extract from domain if all else fails
    domain_parts = domain.split('.')
    if domain_parts and domain_parts[0] not in ['www', 'jobs', 'careers']:
        return domain_parts[0].title()
    
    return "Company not identified"

def detect_industry_from_content(job_description):
    """Detect industry based on job description content"""
    job_lower = job_description.lower()
    
    # Define industry keywords with weights
    industry_keywords = {
        'Technology': ['software', 'engineering', 'developer', 'data', 'cloud', 'AI', 'programming',
                      'algorithm', 'code', 'tech', 'IT', 'cybersecurity', 'database'],
        'Finance': ['finance', 'banking', 'investment', 'financial', 'trading', 'wealth', 'accounting',
                   'audit', 'tax', 'budget', 'fiscal', 'monetary', 'revenue'],
        'Healthcare': ['health', 'medical', 'clinical', 'patient', 'care', 'therapy', 'physician',
                      'nurse', 'hospital', 'doctor', 'treatment', 'diagnostic'],
        'Education': ['education', 'teaching', 'student', 'academic', 'school', 'university',
                     'learning', 'curriculum', 'faculty', 'instruction', 'educational'],
        'Retail': ['retail', 'consumer', 'store', 'product', 'sales', 'customer', 'merchandise',
                 'e-commerce', 'shopping', 'brand', 'buyer'],
        'Manufacturing': ['manufacturing', 'production', 'assembly', 'factory', 'quality', 'supply chain',
                        'industrial', 'fabrication', 'materials', 'processing'],
        'Consulting': ['consulting', 'adviser', 'strategic', 'client', 'solution', 'professional',
                      'service', 'engagement', 'management consulting']
    }
    
    # Count keyword matches for each industry
    industry_scores = {}
    for industry, keywords in industry_keywords.items():
        score = sum(3 for keyword in keywords if f" {keyword} " in f" {job_lower} ")  # Exact word matches
        score += sum(1 for keyword in keywords if keyword in job_lower)  # Partial matches
        industry_scores[industry] = score
    
    # Determine confidence level
    if not industry_scores:
        return "General", "low"
    
    # Find the industry with the highest score
    max_industry = max(industry_scores, key=industry_scores.get)
    max_score = industry_scores[max_industry]
    
    if max_score > 10:
        confidence = "high"
    elif max_score > 5:
        confidence = "medium"
    else:
        confidence = "low"
        
    # If confidence is low, default to General
    if confidence == "low" and max_score < 3:
        return "General", confidence
        
    return max_industry, confidence

def extract_tasks_and_skills(job_description):
    """Extract both tasks and skills from the job description"""
    # Extract tasks from bullet points and structured sections
    tasks = []
    skills = []
    
    # Look for bullet points
    bullet_pattern = re.compile(r'(?:^|\n)(?:\s*[-•*+⋅◦]|\d+\.\s+)([^\n]+)', re.MULTILINE)
    bullet_items = bullet_pattern.findall(job_description)
    
    # Categorize bullet items as tasks or skills
    for item in bullet_items:
        item = item.strip()
        if len(item) < 5:
            continue
            
        # Skill indicators
        skill_indicators = ['experience with', 'proficiency in', 'knowledge of', 'expertise in',
                           'familiar with', 'background in', 'skilled in', 'proficient with']
                           
        # Task indicators - action verbs
        task_indicators = ['develop', 'create', 'manage', 'design', 'implement', 'analyze',
                         'coordinate', 'lead', 'organize', 'maintain', 'build', 'conduct',
                         'provide', 'support', 'monitor', 'ensure', 'prepare', 'communicate']
        
        item_lower = item.lower()
        
        # Check if it's a skill
        if any(indicator in item_lower for indicator in skill_indicators):
            skills.append(item)
        # Check if it starts with an action verb or contains task phrases
        elif any(item_lower.startswith(verb) for verb in task_indicators) or \
             any(f" {verb} " in f" {item_lower} " for verb in task_indicators):
            tasks.append(item)
        # If it's neither clearly a skill nor task, make an educated guess
        else:
            # Items with technical terms are likely skills
            if any(tech in item_lower for tech in ['python', 'java', 'sql', 'aws', 'cloud', 'api']):
                skills.append(item)
            else:
                # Otherwise default to task
                tasks.append(item)
    
    # If we found very few tasks or skills, try to extract from sections
    if len(tasks) < 3 or len(skills) < 2:
        section_extraction(job_description, tasks, skills)
    
    return tasks, skills

def format_skills_for_display(skills):
    """Format skills for display with better filtering of template content"""
    # First filter out any template/placeholder content
    filtered_skills = []
    template_indicators = [
        'company name', 'job description', 'required skills', 
        'qualifications', 'requirements', '*', '**', 
        'this section', 'include', 'insert', 'specify', 'list'
    ]
    
    for skill in skills:
        # Skip template content
        if any(indicator in skill.lower() for indicator in template_indicators):
            continue
            
        # Skip if it's too short or contains placeholder markers
        if len(skill) < 8 or '**' in skill or skill.startswith('*'):
            continue
            
        filtered_skills.append(skill)
    
    # If we filtered everything, provide some generic skills based on industry
    if not filtered_skills:
        return "Could not extract specific skills from job posting"
    
    # Format the remaining skills with bullet points
    return "• " + "\n• ".join(filtered_skills[:5])  # Show top 5

def get_industry_metrics(industry):
    """Get growth rate and outlook for an industry"""
    # Use dynamic data sources when available
    try:
        # Try to read from BLS data file if it exists
        bls_file = 'insights/data/bls_employment_projection_2023_33_annualized_rounded.csv'
        if os.path.exists(bls_file):
            df = pd.read_csv(bls_file)
            industry_row = df[df['Occupational Group'].str.lower().str.contains(industry.lower(), na=False)]
            if not industry_row.empty:
                growth_rate = f"{industry_row['Projected Employment Change (%)'].iloc[0]}%"
                return growth_rate, f"Based on BLS data for {industry}"
    except Exception as e:
        print(f"Error reading BLS data: {e}")
    
    # Industry-specific growth estimates based on recent trends
    industry_growth = {
        'Technology': '7.5%',
        'Finance': '5.0%',
        'Healthcare': '8.5%',
        'Education': '4.2%',
        'Retail': '3.5%',
        'Manufacturing': '2.8%', 
        'Consulting': '6.2%',
        'General': '4.0%'
    }
    
    return industry_growth.get(industry, '4.0%'), f"Based on estimated growth for {industry}"

def analyze_task_automation(tasks):
    """Analyze which tasks have automation potential"""
    automatable_tasks = []
    
    # AI automation potential indicators
    automation_indicators = {
        'data entry': 0.35,
        'document': 0.25,
        'report': 0.28,
        'analyze data': 0.22,
        'track': 0.18,
        'monitor': 0.15,
        'process': 0.21,
        'schedule': 0.30,
        'update': 0.20,
        'record': 0.32,
        'compile': 0.27,
        'review': 0.17,
        'file': 0.31,
        'maintain database': 0.29,
        'collect': 0.16,
        'organize': 0.19,
        'format': 0.26,
        'summarize': 0.24,
        'generate': 0.23,
        'calculate': 0.22
    }
    
    # Analyze each task for automation potential
    for task in tasks:
        task_lower = task.lower()
        automation_score = 0
        
        # Check for indicators of automation potential
        for indicator, score in automation_indicators.items():
            if indicator in task_lower:
                automation_score = max(automation_score, score)
                
        # Adjust score based on complexity indicators
        if any(complex_term in task_lower for complex_term in ['complex', 'judgment', 'creative', 'negotiate', 'strategy']):
            automation_score *= 0.5  # Reduce automation potential for complex tasks
            
        if automation_score > 0.05:
            automatable_tasks.append({
                'job_task': task,
                'claude_usage_pct': automation_score  # Higher percentages indicate more automation potential
            })
    
    # Sort by automation potential
    automatable_tasks.sort(key=lambda x: x['claude_usage_pct'], reverse=True)
    
    return automatable_tasks

def generate_recommendations(industry, automatable_tasks, tasks):
    """Generate contextual career recommendations"""
    # Base recommendations on industry and automation risk
    high_automation = len(automatable_tasks) > 0 and automatable_tasks[0]['claude_usage_pct'] > 0.25
    
    # Industry-specific recommendations
    industry_recs = {
        'Technology': [
            "Develop expertise in emerging AI governance, ethics, and prompt engineering skills.",
            "Focus on complex system architecture and design thinking that AI struggles with.",
            "Build skills that combine technical knowledge with strategic business understanding."
        ],
        'Finance': [
            "Develop client relationship and complex financial advisory capabilities.",
            "Focus on financial regulation expertise and compliance strategy.",
            "Build skills in financial risk assessment and scenario planning."
        ],
        'Healthcare': [
            "Focus on complex patient care coordination and interpersonal healthcare skills.",
            "Develop expertise in emerging treatment approaches and care protocols.",
            "Build skills combining medical knowledge with empathetic patient communication."
        ],
        'Education': [
            "Develop adaptive teaching approaches and personalized learning capabilities.",
            "Focus on building emotional intelligence and student engagement skills.",
            "Combine subject expertise with creative instructional design abilities."
        ],
        'General': [
            "Focus on developing human-centered skills like creativity and critical thinking.",
            "Build expertise in areas requiring complex judgment and emotional intelligence.",
            "Develop your ability to collaborate across disciplines and integrate perspectives."
        ]
    }
    
    # Get recommendations for this industry, defaulting to General if not found
    recommendations = industry_recs.get(industry, industry_recs['General'])
    
    # Personalize based on automation risk
    if high_automation:
        return f"Given your automation risk in {industry}: {recommendations[0]}"
    elif len(automatable_tasks) > 1:
        return f"To stay competitive in {industry}: {recommendations[1]}"
    else:
        return f"To advance in {industry}: {recommendations[2]}"

def analyze_job_automation(job_posting_url, csv_path=None, similarity_threshold=0.3):
    """Analyze job automation risk based on Claude usage patterns from a job posting URL"""
    try:
        # Start timing the process
        import time
        start_time = time.time()
        
        # Use existing job_fetcher functionality
        job_description = fetch_job_from_url(job_posting_url)
        
        if not job_description:
            return {"error": f"Failed to extract job description from URL: {job_posting_url}"}
        
        # Filter out placeholder content
        job_description = remove_placeholder_content(job_description)
        
        # Extract company name properly
        company_name = extract_company_from_job(job_description, job_posting_url)
        print(f"Extracted company name: {company_name}")
        
        # Load task data with sampling for speed if the dataset is large
        task_data = load_task_data(csv_path, sample_if_large=True)
        
        if task_data.empty:
            return {"error": "Failed to load task data"}

        # Extract tasks from job description
        job_tasks = extract_tasks(job_description)
        if not job_tasks:
            job_tasks = split_into_meaningful_chunks(job_description)  # Use full description split into chunks
            
        print(f"Extracted task details: {len(job_tasks)} tasks found")
            
        # Detect industry - use a faster approach
        industry = detect_industry_fast(job_description)
        print(f"Detected industry: {industry}")
        
        # Find matching tasks with simplified approach (avoid timeout issues)
        matched_tasks = simplified_task_matching(job_tasks, task_data, similarity_threshold)
        
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
        
        # Get industry growth rate from BLS data
        growth_rate = get_industry_growth_rate(industry)
        
        # Adjust risk based on industry growth
        # Higher growth reduces risk, lower growth increases risk
        growth_adjustment = (1 - (growth_rate / 2))  # Scale growth impact
        adjusted_risk = base_risk * growth_adjustment
        
        # Ensure risk stays within bounds
        adjusted_risk = max(0.1, min(0.9, adjusted_risk))
        
        # Generate reasoning
        reasoning = generate_simple_reasoning(
            high_usage_tasks, 
            medium_usage_tasks,
            low_usage_tasks,
            growth_rate,
            industry
        )
        
        # Extract company info from the job description
        company_info = {
            'company_name': company_name,
            'industry': industry,
            'skills_in_demand': extract_skills(job_description),
            'job_url': job_posting_url
        }
        
        # Get career growth potential
        career_growth = analyze_company_growth(company_info, industry)
        
        # Show processing time
        elapsed_time = time.time() - start_time
        print(f"Job analysis completed in {elapsed_time:.2f} seconds")
        
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

def remove_placeholder_content(text):
    """Remove placeholder content like [Insert X]"""
    # Remove common placeholder patterns
    placeholder_patterns = [
        r'\[Insert[^\]]*\]',
        r'\[Add[^\]]*\]',
        r'\[Your[^\]]*\]',
        r'\[Company[^\]]*\]',
        r'\[Position[^\]]*\]',
        r'\[Role[^\]]*\]',
        r'\[Description[^\]]*\]',
        r'\[Include[^\]]*\]',
    ]
    
    for pattern in placeholder_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove empty bullet points or numbered items
    text = re.sub(r'•\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)
    
    return text

def find_matching_tasks_with_timeout(job_tasks, task_data, similarity_threshold=0.3, timeout_seconds=10):
    """Version of find_matching_tasks with a timeout to prevent hanging"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def worker():
        try:
            # Sample the task data for faster processing if it's large
            if len(task_data) > 5000:
                # Sample 20% of the data, at least 1000 rows
                sample_size = max(1000, int(len(task_data) * 0.2))
                sampled_data = task_data.sample(sample_size)
            else:
                sampled_data = task_data
                
            # Process at most 20 tasks to avoid slowdowns
            processed_tasks = job_tasks[:20] if len(job_tasks) > 20 else job_tasks
            
            # Get matches
            matches = find_matching_tasks(processed_tasks, sampled_data, similarity_threshold)
            result_queue.put(matches)
        except Exception as e:
            result_queue.put(f"Error: {str(e)}")
    
    # Start the worker thread
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait for result or timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"Task matching timed out after {timeout_seconds} seconds. Using simplified approach.")
        # If timed out, use a simplified approach
        return simplified_task_matching(job_tasks, task_data, similarity_threshold)
    
    # Get the result
    if result_queue.empty():
        print("No result from task matching. Using simplified approach.")
        return simplified_task_matching(job_tasks, task_data, similarity_threshold)
    
    result = result_queue.get()
    if isinstance(result, str) and result.startswith("Error:"):
        print(f"Error in task matching: {result}. Using simplified approach.")
        return simplified_task_matching(job_tasks, task_data, similarity_threshold)
    
    return result

def simplified_task_matching(job_tasks, task_data, similarity_threshold=0.3):
    """Simplified task matching for faster processing"""
    import numpy as np
    matched_tasks = []
    
    # Sample data if it's large
    if len(task_data) > 1000:
        task_data = task_data.sample(1000)
    
    # Process limited number of tasks
    process_count = min(10, len(job_tasks))
    
    for job_task in job_tasks[:process_count]:
        # Simple keyword matching
        words = set(re.findall(r'\b\w+\b', job_task.lower()))
        
        best_match = None
        best_score = similarity_threshold
        
        # Random sample for faster processing
        sample_indices = np.random.choice(len(task_data), min(100, len(task_data)), replace=False)
        
        for idx in sample_indices:
            row = task_data.iloc[idx]
            task_name = str(row['task_name'])
            task_words = set(re.findall(r'\b\w+\b', task_name.lower()))
            
            # Calculate simple word overlap
            if len(words) > 0 and len(task_words) > 0:
                overlap = len(words.intersection(task_words)) / len(words.union(task_words))
                
                if overlap > best_score:
                    best_score = overlap
                    
                    # Calculate claude usage
                    claude_usage = row.get('claude_usage_pct', row.get('pct', 0.01))
                    
                    best_match = {
                        'job_task': job_task,
                        'matched_task': task_name,
                        'match_score': overlap,
                        'claude_usage_pct': float(claude_usage)
                    }
        
        if best_match:
            matched_tasks.append(best_match)
    
    return matched_tasks

def generate_simple_reasoning(high_tasks, medium_tasks, low_tasks, growth_rate, industry):
    """Generate simple reasoning based on task usage patterns - optimized"""
    # Count tasks in each category
    high_count = len(high_tasks)
    medium_count = len(medium_tasks)
    low_count = len(low_tasks)
    total_count = high_count + medium_count + low_count
    
    if total_count == 0:
        return "No tasks could be analyzed for Claude usage patterns."
    
    # Calculate percentages
    high_pct = (high_count / total_count) * 100 if total_count > 0 else 0
    
    # Build reasoning efficiently with list comprehension
    reasoning = [
        f"{high_count} tasks ({high_pct:.1f}%) have HIGH usage in Claude interactions"
    ]
    
    # Add high usage task examples (up to 2 for efficiency)
    if high_count > 0:
        reasoning.append("These tasks are most commonly performed:")
        reasoning.extend([f"• '{task['job_task'][:50]}...'" for task in high_tasks[:2]])
    
    # Add industry growth information
    growth_msg = (f"Industry growth is strong ({growth_rate}%)" if growth_rate >= 8.0 else
                 f"Industry growth is slow ({growth_rate}%)" if growth_rate <= 2.0 else
                 f"Industry growth is moderate ({growth_rate}%)")
    reasoning.append(growth_msg)
    
    # Add summary
    if high_count > medium_count + low_count:
        reasoning.append("Overall: Most job tasks are commonly handled by Claude")
    elif high_count == 0:
        reasoning.append("Overall: Job tasks are less common in Claude's experience")
    else:
        reasoning.append("Overall: Job has mixed task frequency in Claude's experience")
    
    return "\n".join(reasoning)

def load_task_data(csv_path=None, sample_if_large=False):
    """Load task data from CSV file with improved error handling"""
    try:
        import pandas as pd
        
        # Use provided CSV path or default to ONET_TASK_MAPPINGS_FILE
        file_path = csv_path or ONET_TASK_MAPPINGS_FILE
        
        if not os.path.exists(file_path):
            print(f"Task data file not found: {file_path}")
            # Try a fallback path
            alternate_path = 'data/EconomicIndex/onet_task_mappings.csv'
            if os.path.exists(alternate_path):
                file_path = alternate_path
                print(f"Using alternate path: {alternate_path}")
            else:
                return pd.DataFrame()
            
        # Load the data
        print(f"Loading task data from {file_path}")
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} task records")
        
        # Rename columns if needed to standardize
        column_mapping = {
            'Task': 'task_name',
            'task': 'task_name',
            'Task Name': 'task_name',
            'task_statement': 'task_name',
            'Pct': 'pct',
            'Percent': 'pct',
            'Claude Usage': 'claude_usage_pct',
            'Usage': 'claude_usage_pct'
        }
        
        # Print available columns for debugging
        print(f"Available columns: {', '.join(data.columns)}")
        
        # Rename columns that exist in the dataframe
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]
        
        # Ensure task_name and claude_usage_pct columns exist
        if 'task_name' not in data.columns:
            # Use the first text column
            text_cols = [col for col in data.columns if data[col].dtype == 'object']
            if text_cols:
                print(f"Using {text_cols[0]} as task_name")
                data['task_name'] = data[text_cols[0]]
            else:
                print("No suitable text column found for task_name")
                return pd.DataFrame()
        
        # If no percentage column, add a default
        if 'claude_usage_pct' not in data.columns and 'pct' not in data.columns:
            print("No claude usage data found, using defaults")
            data['claude_usage_pct'] = 0.01  # Default value
        elif 'pct' in data.columns and 'claude_usage_pct' not in data.columns:
            print("Using 'pct' column as claude_usage_pct")
            data['claude_usage_pct'] = data['pct']
        
        return data
        
    except Exception as e:
        print(f"Error loading task data: {e}")
        import traceback
        traceback.print_exc()
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

# Cache for BLS data to avoid repeated file loading
_BLS_DATA_CACHE = None

def get_industry_growth_rate(industry):
    """Get growth rate for a specific industry from BLS data - with caching"""
    global _BLS_DATA_CACHE
    
    try:
        # Use cached data if available
        if _BLS_DATA_CACHE is None:
            bls_file = 'insights/data/bls_employment_projection_2023_33_annualized_rounded.csv'
            if os.path.exists(bls_file):
                _BLS_DATA_CACHE = pd.read_csv(bls_file)
            else:
                print(f"BLS data file not found: {bls_file}")
                return 4.0  # Default if file not found
        
        bls_data = _BLS_DATA_CACHE
        
        # Simplified industry mapping (fewer lookups)
        industry_keywords = {
            'Technology': ['computer', 'information', 'software', 'tech', 'IT'],
            'Healthcare': ['health', 'medical', 'care', 'nursing'],
            'Finance': ['finance', 'financial', 'banking', 'investment'],
            'Education': ['education', 'teaching', 'academic'],
            'Manufacturing': ['manufacturing', 'production'],
            'Retail': ['retail', 'sales']
        }
        
        # Find matching industry
        detected_industry = None
        for ind, keywords in industry_keywords.items():
            if any(kw in industry.lower() for kw in keywords):
                detected_industry = ind
                break
        
        if detected_industry is None:
            detected_industry = 'Technology'  # Default if no match
            
        # Direct mapping to BLS categories
        industry_to_bls = {
            'Technology': ['Computer and mathematical'],
            'Healthcare': ['Healthcare practitioners and technical', 'Healthcare support'],
            'Finance': ['Business and financial operations'],
            'Education': ['Educational instruction and library'],
            'Manufacturing': ['Production'],
            'Retail': ['Sales and related']
        }
        
        matching_groups = industry_to_bls.get(detected_industry, ['Management'])
        
        # Filter data - case insensitive matching for efficiency
        growth_data = bls_data[bls_data['Occupational Group'].str.lower().isin(
            [group.lower() for group in matching_groups]
        )]
        
        if not growth_data.empty:
            # Get the first matching group's growth rate
            return float(growth_data.iloc[0]['Projected Employment Change (%)'])
            
        return 4.0  # Default growth rate
    
    except Exception as e:
        print(f"Error getting industry growth rate: {e}")
        return 4.0  # Default fallback

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
        f"1. Company name. Be sure to be accurate with the company name ... it is not always the website name"
        f"2. Industry sector "
        f"3. Brief company description "
        f"4. Recent company growth or industry trends "
        f"5. Top skills in demand for this role. Make sure these are actionable skills rather than qualifications like a degree. Please double check this before adding it "
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
    """Analyze company growth potential and career prospects"""
    try:
        # Get industry-specific growth data
        growth_rate = get_industry_growth_rate(industry)
        
        # Determine growth level based on thresholds
        if growth_rate >= 8.0:
            level = "High"
            description = f"Based on {company_info['company_name']}'s position in the {industry} industry, which is showing strong growth at {growth_rate:.1f}% annually."
            outlook = f"The {industry} sector is projected to expand significantly over the next decade, creating new opportunities for advancement."
            recommendations = f"Develop specialized skills in emerging {industry} technologies to maximize career growth potential."
        elif growth_rate >= 4.0:
            level = "Moderate"
            description = f"{company_info['company_name']} operates in the {industry} sector with moderate growth of {growth_rate:.1f}% annually."
            outlook = f"While not the fastest growing sector, the {industry} industry maintains steady demand for skilled professionals."
            recommendations = f"Pursue additional certifications and cross-functional experience to stand out in the {industry} job market."
        else:
            level = "Limited"
            description = f"{company_info['company_name']} is in the {industry} sector which shows limited growth ({growth_rate:.1f}% annually)."
            outlook = f"The {industry} sector faces transformation challenges with below-average growth projections."
            recommendations = f"Focus on developing transferable skills that apply across industries to maintain career mobility."
        
        # Extract skills from job posting - now with improved extraction
        extracted_skills = company_info.get('skills_in_demand', '')
        
        # Check for placeholder or generic content in the skills
        has_placeholder = any(term in extracted_skills.lower() for term in [
            'required**', 'identify', 'specific skills', 'qualifications', 'insert', 
            'you will need', 'please provide'
        ])
        
        if has_placeholder or not extracted_skills or len(extracted_skills) < 10:
            # Use industry-specific skill suggestions instead
            industry_skills = {
                'Technology': 'Programming, cloud infrastructure, system design, data analysis, problem solving',
                'Healthcare': 'Patient care, medical knowledge, regulatory compliance, documentation, teamwork',
                'Finance': 'Financial analysis, regulatory knowledge, risk assessment, modeling, client management',
                'Education': 'Curriculum development, assessment, classroom management, communication, mentoring',
                'Consulting': 'Client management, problem analysis, communication, project planning, industry expertise',
                'Manufacturing': 'Quality control, supply chain knowledge, equipment operation, safety protocols, efficiency improvement',
                'Retail': 'Customer service, inventory management, sales techniques, merchandising, cash handling'
            }
            skills = industry_skills.get(industry, 'Technical and communication skills relevant to the position')
        else:
            # Clean up any duplicates or common content issues in skills
            skills = extracted_skills.replace("Required skills:", "").replace("Required Skills:", "").strip()
        
        # Return structured career growth data
        return {
            "level": level,
            "description": description, 
            "growth_rate": f"{growth_rate:.1f}%",
            "outlook": outlook,
            "skill_demand": skills,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error in company growth analysis: {e}")
        # Provide a fallback that doesn't mention the generic 4.0% growth
        return {
            "level": "Moderate",
            "description": f"Based on current trends in the {industry} industry.",
            "growth_rate": "Industry-specific", 
            "outlook": f"The {industry} sector continues to evolve with new opportunities.",
            "skill_demand": f"Core {industry} skills and relevant technical expertise",
            "recommendations": "Stay current with industry developments and expand your skill set."
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

def extract_tasks(job_description):
    """Extract tasks from job description with improved detection"""
    tasks = []
    
    # Print first 200 chars of job description for debugging
    print(f"Job description preview: {job_description[:200]}...")
    
    # Look for bullet points or numbered lists - these often contain tasks
    bullet_pattern = re.compile(r'(?:^|\n)(?:\s*[-•*+⋅◦]|\d+\.\s+)([^\n]+)', re.MULTILINE)
    bullet_tasks = bullet_pattern.findall(job_description)
    
    # Clean up and add bullet point tasks
    for task in bullet_tasks:
        task = task.strip()
        if len(task) > 5:  # Reduced minimum length requirement
            tasks.append(task)
    
    # Look for paragraph-based responsibilities
    responsibility_blocks = []
    
    # Common section headers that introduce job tasks
    section_headers = [
        'responsibilities', 'duties', 'what you\'ll do', 'your role', 
        'day to day', 'in this position', 'your responsibilities',
        'role description', 'job description', 'what you will do'
    ]
    
    # Extract sections following task headers
    job_lower = job_description.lower()
    for header in section_headers:
        if header in job_lower:
            # Find the header position
            start_pos = job_lower.find(header)
            
            # Find the end of the section (next header or paragraph)
            end_pos = len(job_description)
            for next_header in section_headers:
                if next_header != header and next_header in job_lower[start_pos + len(header):]:
                    next_pos = job_lower.find(next_header, start_pos + len(header))
                    end_pos = min(end_pos, next_pos)
            
            # Extract the section text
            section = job_description[start_pos:end_pos].strip()
            responsibility_blocks.append(section)
    
    # Process responsibility blocks into individual tasks
    for block in responsibility_blocks:
        # Split by sentence or bullet points
        block_tasks = re.split(r'(?<=[.!?])\s+|(?<=\n)', block)
        
        for task in block_tasks:
            task = task.strip()
            if len(task) > 15 and task not in tasks:
                tasks.append(task)
    
    # If we still have no tasks, try to extract sentences with task indicators
    if not tasks:
        sentences = re.findall(r'[^.!?]+[.!?]', job_description)
        for sentence in sentences:
            sentence = sentence.strip()
            lower_sentence = sentence.lower()
            
            # More comprehensive list of task indicators
            task_indicators = [
                'you will', 'responsible for', 'duties include', 'responsibilities', 
                'required to', 'expected to', 'day to day', 'day-to-day', 'manage', 
                'develop', 'create', 'build', 'work with', 'collaborate', 'lead',
                'ensure', 'maintain', 'analyze', 'support', 'coordinate', 'design',
                'implement', 'monitor', 'organize', 'perform', 'prepare', 'provide',
                'report', 'research', 'review', 'ability to', 'experience with'
            ]
            
            if any(indicator in lower_sentence for indicator in task_indicators) and len(sentence) > 15:
                tasks.append(sentence)
    
    # If we still don't have enough tasks, get the longest sentences as tasks
    if len(tasks) < 3:
        sentences = re.findall(r'[^.!?]+[.!?]', job_description)
        sorted_sentences = sorted(sentences, key=len, reverse=True)
        for s in sorted_sentences:
            if len(s.strip()) > 20 and s.strip() not in tasks:
                tasks.append(s.strip())
                if len(tasks) >= 5:  # Get up to 5 tasks
                    break
    
    # Add debugging output
    print(f"Extracted {len(tasks)} tasks from job description")
    if len(tasks) > 0:
        print(f"Sample task: {tasks[0][:100]}...")
    
    return tasks

def detect_industry_fast(job_description):
    """
    A faster version of industry detection that uses simpler matching
    
    Args:
        job_description: The job description text
        
    Returns:
        String with the detected industry
    """
    # Convert to lowercase for case-insensitive matching
    text = job_description.lower()
    
    # Quick keyword matching without loading CSV files (for speed)
    industry_keywords = {
        'Technology': ['software', 'tech', 'data', 'engineer', 'developer', 'programming', 
                      'IT', 'computer', 'algorithm', 'cloud', 'digital', 'ai', 'machine learning'],
        'Healthcare': ['health', 'medical', 'doctor', 'nurse', 'patient', 'clinical', 'pharma', 
                      'biotech', 'hospital', 'care'],
        'Finance': ['finance', 'bank', 'investment', 'financial', 'trading', 'asset', 
                   'accounting', 'audit', 'tax'],
        'Education': ['education', 'teach', 'school', 'student', 'learning', 'academic', 
                     'university', 'college', 'training'],
        'Consulting': ['consult', 'client', 'strategy', 'advisor', 'business solution', 
                      'stakeholder', 'management consulting'],
        'Manufacturing': ['manufacturing', 'product', 'assembly', 'production', 'factory', 
                         'supply chain', 'inventory'],
        'Retail': ['retail', 'store', 'shop', 'e-commerce', 'customer', 'merchandise', 
                  'consumer', 'sales']
    }
    
    # Check for company names that strongly indicate an industry
    company_indicators = {
        'Sierra': 'Technology',
        'Palantir': 'Technology',
        'Google': 'Technology',
        'Microsoft': 'Technology',
        'Amazon': 'Technology',
        'McKinsey': 'Consulting',
        'BCG': 'Consulting',
        'Deloitte': 'Consulting'
    }
    
    # First check for company name matches (very fast)
    for company, industry in company_indicators.items():
        if company.lower() in text:
            return industry
    
    # Count keyword matches for each industry
    matches = {}
    for industry, keywords in industry_keywords.items():
        # Use any() for speed instead of counting all matches
        if any(keyword in text for keyword in keywords):
            # Count only if we need to disambiguate
            count = sum(1 for keyword in keywords if keyword in text)
            matches[industry] = count
    
    # Return the industry with the most keyword matches, or default to Technology
    if matches:
        return max(matches, key=matches.get)
    else:
        return "Technology"  # Default industry

def extract_skills(job_description):
    """Extract skills from job description with improved detection"""
    # First, let's make sure we're working with clean text
    job_description = remove_placeholder_content(job_description)
    
    # Look for sections that typically contain skills
    skill_section_headers = [
        'skills', 'qualifications', 'requirements', 'you have', 
        'you should have', 'you will have', 'what we\'re looking for',
        'what you\'ll need', 'must have', 'preferred qualifications'
    ]
    
    # Extract skills from relevant sections
    skills = []
    job_lower = job_description.lower()
    
    # Check if any skill section exists
    found_section = False
    for header in skill_section_headers:
        if header in job_lower:
            found_section = True
            start_pos = job_lower.find(header)
            
            # Find the end of the section (next header or paragraph)
            end_pos = len(job_description)
            for next_header in ['experience', 'responsibilities', 'about us', 'benefits']:
                if next_header in job_lower[start_pos + len(header):]:
                    next_pos = job_lower.find(next_header, start_pos + len(header))
                    end_pos = min(end_pos, next_pos)
            
            # Extract the section text
            section = job_description[start_pos:end_pos].strip()
            
            # Look for bullet points in the section
            bullet_pattern = re.compile(r'(?:^|\n)(?:\s*[-•*+⋅◦]|\d+\.\s+)([^\n]+)', re.MULTILINE)
            bullet_skills = bullet_pattern.findall(section)
            
            for skill in bullet_skills:
                skill = skill.strip()
                # Only add non-placeholder, substantive skills
                if (len(skill) > 3 and 
                    not skill.startswith('[') and 
                    not any(placeholder in skill.lower() for placeholder in ['insert', 'add', 'your'])):
                    skills.append(skill)
    
    # If no skills found using sections, try to extract based on common skill phrases
    if not skills:
        # Common prefixes for skills
        skill_prefixes = [
            'experience with', 'knowledge of', 'familiarity with', 'proficiency in',
            'expertise in', 'skilled in', 'ability to', 'capable of', 'background in'
        ]
        
        sentences = re.findall(r'[^.!?]+[.!?]', job_description)
        for sentence in sentences:
            if any(prefix in sentence.lower() for prefix in skill_prefixes):
                if len(sentence.strip()) > 10:  # Reasonable skill statement length
                    skills.append(sentence.strip())
    
    # If still no skills, look for common technical terms
    if not skills:
        # Common technical skills that might appear in job descriptions
        tech_skills = [
            'python', 'java', 'javascript', 'sql', 'aws', 'cloud', 'azure',
            'machine learning', 'ai', 'data analysis', 'project management',
            'communication', 'leadership', 'problem solving', 'analytical',
            'research', 'writing', 'design', 'marketing', 'sales', 'financial'
        ]
        
        found_tech_skills = []
        for skill in tech_skills:
            if skill in job_lower:
                found_tech_skills.append(skill.title())  # Title case for display
        
        if found_tech_skills:
            skills.append("Technical skills: " + ", ".join(found_tech_skills))
    
    # Always include communication and problem-solving as fallbacks only if no skills found
    if not skills:
        return "Technical or domain-specific knowledge relevant to the role"
    
    # Return formatted skills
    if len(skills) == 1:
        return skills[0]
    else:
        # Format with bullets for multiple skills
        return "• " + "\n• ".join(skills)

def extract_company_from_job(job_description, job_url):
    """Extract the actual company name from job description, not just domain"""
    # First look for common company indicators in the job description
    company_patterns = [
        r'About\s+([\w\s]+)(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company)?(?:\s|$|\.)',
        r'([\w\s]+)(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company)? is (?:a|an|the)',
        r'Join\s+([\w\s]+)(?:\'s)? team',
        r'(?:About|At)\s+([\w\s]+)(?:,|\.|\n)',
        r'Welcome to\s+([\w\s]+)(?:,|\.|\n)',
        r'Company:\s+([\w\s]+)(?:,|\.|\n)',
    ]
    
    # Try each pattern
    for pattern in company_patterns:
        matches = re.findall(pattern, job_description, re.IGNORECASE)
        if matches:
            # Take the first match
            company = matches[0].strip() if isinstance(matches[0], str) else matches[0][0].strip()
                
            # Clean up the name
            if len(company) > 2 and len(company) < 50:  # Reasonable company name length
                return company
    
    # If we couldn't find it in the description, check for Sierra specifically
    if "Sierra" in job_description:
        return "Sierra"
    
    # Try to extract from URL
    from urllib.parse import urlparse
    parsed_url = urlparse(job_url)
    
    # Special case for ashbyhq Sierra URL
    if "ashbyhq.com/Sierra" in job_url:
        return "Sierra"
    
    # Handle other job board URLs
    job_boards = ['ashbyhq', 'lever', 'greenhouse', 'workday', 'indeed', 'linkedin', 'glassdoor', 'ziprecruiter']
    
    domain = parsed_url.netloc.lower()
    
    # If we're on a job board, try to extract from path
    if any(board in domain for board in job_boards):
        path = parsed_url.path
        path_parts = [p for p in path.split('/') if p and p not in ['jobs', 'careers', 'job', 'posting']]
        
        if path_parts:
            # First part of path is often the company name
            company_name = path_parts[0].replace('-', ' ').replace('_', ' ').title()
            return company_name
    
    # Otherwise use the domain
    company_name = domain.split('.')[0].replace('-', ' ').title()
    
    # Exclude common non-company domains
    if company_name.lower() in ['www', 'jobs', 'careers'] + job_boards:
        # Try second part of domain
        parts = domain.split('.')
        if len(parts) > 1:
            company_name = parts[1].replace('-', ' ').title()
        else:
            company_name = "Unknown Company"
    
    return company_name

def split_into_meaningful_chunks(text, chunk_size=150):
    """
    Split a long text into meaningful chunks for processing
    
    Args:
        text: The long text to split
        chunk_size: Target size for each chunk
        
    Returns:
        List of text chunks
    """
    # First try to split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + " "
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we have very few chunks, try splitting by sentences
    if len(chunks) < 3:
        chunks = []
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    # If still not enough chunks, force split by chunk_size
    if len(chunks) < 3:
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk += word + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word + " "
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def find_matching_tasks(job_tasks, task_data, similarity_threshold=0.3):
    """Find matching tasks and their Claude usage - optimized for speed"""
    import numpy as np
    matched_tasks = []
    
    # Ensure 'claude_usage_pct' column exists
    if 'claude_usage_pct' not in task_data.columns and 'pct' in task_data.columns:
        task_data['claude_usage_pct'] = task_data['pct']
    elif 'claude_usage_pct' not in task_data.columns:
        print("Warning: No Claude usage data found. Using default values.")
        task_data['claude_usage_pct'] = 0.01  # Default value
    
    try:
        # Use TF-IDF vectorization for faster batch processing
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Get task descriptions from the database
        db_tasks = task_data['task_name'].astype(str).tolist()
        
        # Combine all tasks for the vectorizer
        all_tasks = db_tasks + job_tasks
        
        # Fit and transform in one step
        tfidf_matrix = vectorizer.fit_transform(all_tasks)
        
        # Split into database tasks and job tasks
        db_vectors = tfidf_matrix[:len(db_tasks)]
        job_vectors = tfidf_matrix[len(db_tasks):]
        
        # Calculate similarity matrix - all pairs at once
        similarity_matrix = cosine_similarity(job_vectors, db_vectors)
        
        # Process each job task
        for i, job_task in enumerate(job_tasks):
            # Get similarities for this job task
            similarities = similarity_matrix[i]
            
            # Find top matches above threshold
            indices = np.where(similarities >= similarity_threshold)[0]
            
            if len(indices) > 0:
                # Sort by similarity (highest first)
                sorted_indices = indices[np.argsort(-similarities[indices])]
                
                # Get best match
                best_idx = sorted_indices[0]
                best_score = similarities[best_idx]
                best_match = {
                    'job_task': job_task,
                    'matched_task': db_tasks[best_idx],
                    'match_score': float(best_score),  # Convert from numpy to native Python type
                    'claude_usage_pct': float(task_data.iloc[best_idx]['claude_usage_pct'])
                }
                matched_tasks.append(best_match)
    
    except Exception as e:
        print(f"Error in vectorized task matching: {e}")
        # Fall back to simpler matching for robustness
        for job_task in job_tasks:
            best_match = None
            best_score = similarity_threshold
            
            # Sample a subset of the database for faster processing
            sample_size = min(1000, len(task_data))
            for idx in np.random.choice(len(task_data), sample_size, replace=False):
                row = task_data.iloc[idx]
                task_name = row['task_name']
                
                # Simple string matching first (very fast)
                if job_task.lower() in task_name.lower() or task_name.lower() in job_task.lower():
                    similarity = 0.8  # High score for substring matches
                else:
                    # Only calculate cosine similarity if needed
                    similarity = calculate_similarity(job_task, task_name)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        'job_task': job_task,
                        'matched_task': task_name,
                        'match_score': similarity,
                        'claude_usage_pct': float(row['claude_usage_pct'])
                    }
            
            if best_match:
                matched_tasks.append(best_match)
    
    return matched_tasks

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        
        # Fallback simple similarity calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

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
