import os
import requests
import json
import re
from dotenv import load_dotenv
from config import ECONOMIC_DATA_FILE, BLS_EMPLOYMENT_FILE, ONET_TASK_MAPPINGS_FILE

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def fetch_job_from_url(url):
    """Fetch job description from a URL"""
    try:
        # First try with Perplexity API if available
        if PERPLEXITY_API_KEY:
            try:
                return fetch_with_perplexity(url)
            except Exception as e:
                print(f"Perplexity fetch failed: {e}. Trying backup method...")
        
        # Backup method using requests
        print("Using direct request to fetch job posting...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        
        # Extract main content from HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Try to find job description container (common patterns)
        job_container = None
        for selector in ['.job-description', '.description', '[data-job-description]', 
                          '#job-details', '.job-details', '[class*="job"][class*="description"]']:
            container = soup.select_one(selector)
            if container:
                job_container = container
                break
        
        # If no specific container found, use main content
        if not job_container:
            job_container = soup.find('main') or soup.find('body')
        
        text = job_container.get_text(separator='\n')
        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch URL: {e}")
    except Exception as e:
        raise Exception(f"Error processing job page: {e}")

def fetch_with_perplexity(url):
    """Fetch job description from a URL using Perplexity API"""
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key not found. Check your .env file.")
    
    print(f"Fetching job description from URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Improved prompt to focus on job responsibilities and skills
    payload = {
        "model": "sonar",  # Using the base sonar model which has web browsing capability
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized job analyzer focusing on extracting specific components from job postings. Your primary task is to extract ONLY the following sections from the job posting at the provided URL:\n\n1. Job responsibilities and tasks (what the person will actually do)\n2. Required skills and qualifications\n3. Job title and role description\n\nIgnore metadata like location, salary, company benefits, application procedures, etc. Format the output as clear sections with bullet points for responsibilities and skills. Focus on actionable job tasks that describe the actual work performed."
            },
            {
                "role": "user",
                "content": f"Extract only the job responsibilities, tasks, required skills, and role description from this job posting: {url}"
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.9,
        "search_domain_filter": None,
        "return_images": False, 
        "return_related_questions": False,
        "stream": False
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", 
                               headers=headers, 
                               json=payload,
                               timeout=30)  # Longer timeout for web scraping
        
        if response.status_code == 200:
            result = response.json()
            # Parse response
            job_description = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if job_description:
                print(f"Successfully fetched job description ({len(job_description)} characters)")
                # Additional processing to clean up the response
                return clean_job_description(job_description)
            else:
                raise ValueError("Empty response received from Perplexity API")
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        print(f"Error fetching job from URL: {e}")
        raise

def clean_job_description(text):
    """Clean up the job description to focus on tasks and skills"""
    # Remove any mentions of "I couldn't find" or similar phrases
    phrases_to_remove = [
        "I couldn't find", 
        "I could not find",
        "I didn't find",
        "I was unable to",
        "doesn't provide",
        "does not provide"
    ]
    
    for phrase in phrases_to_remove:
        if phrase.lower() in text.lower():
            parts = re.split(f"(?i){re.escape(phrase)}[^.]*\\.", text, 1)
            if len(parts) > 1:
                text = parts[1].strip()
    
    # Remove any lines that don't describe tasks or skills
    lines = text.split('\n')
    filtered_lines = []
    
    metadata_markers = [
        "location:", 
        "salary:", 
        "type:", 
        "posted", 
        "apply", 
        "benefits:",
        "about us",
        "about the company",
        "cookie policy",
        "privacy notice"
    ]
    
    important_sections = [
        "what we do",
        "responsibilities",
        "requirements", 
        "qualifications", 
        "skills",
        "what you'll do",
        "you'll",
        "you will",
        "expect to",
        "job description"
    ]
    
    is_relevant_section = True
    
    for line in lines:
        line_lower = line.lower()
        
        # Skip privacy notices and other irrelevant content
        if "cookie" in line_lower or "privacy" in line_lower:
            continue
            
        # Skip lines with metadata
        if any(marker in line_lower for marker in metadata_markers):
            is_relevant_section = False
            continue
            
        # Look for section headers that indicate we're back to relevant content
        if any(marker in line_lower for marker in important_sections):
            is_relevant_section = True
            filtered_lines.append(line)  # Include the section header
            continue
            
        # For Palantir specifically, detect the bullet points section
        if "No two days are the same" in line or "you can expect to" in line_lower:
            is_relevant_section = True
            filtered_lines.append(line)
            continue
            
        # Add line if in relevant section and it's not empty
        if is_relevant_section and line.strip():
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines) 