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
    """
    Simple job fetcher that gets job description from a URL.
    Returns the text description with minimal processing.
    """
    try:
        if PERPLEXITY_API_KEY:
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Extract the job description from this URL: {url}. Return ONLY the job description text, company name, and required skills. No commentary or formatting."
                    }
                ],
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                job_description = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return job_description
            else:
                raise Exception(f"API error: {response.status_code}")
            
        # Fallback to direct request
        response = requests.get(url, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        
        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
        
    except Exception as e:
        print(f"Error fetching job: {e}")
        return ""

def fetch_with_perplexity(url):
    """Fetch job description from a URL using Perplexity API - improved for better content extraction"""
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key not found. Check your .env file.")
    
    print(f"Fetching job description from URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Much more specific prompt to avoid getting templates
    payload = {
        "model": "sonar",  # Using the base sonar model which has web browsing capability
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized job data extractor. Extract ONLY the ACTUAL content from job listings - no templates, no placeholders, no '[Insert X]' text. If you can't find specific information, leave it out rather than using placeholders. Focus on concrete job responsibilities, skills, and company information actually stated in the posting."
            },
            {
                "role": "user",
                "content": f"Extract ONLY the concrete, specific job details from this URL: {url}\n\nPlease include:\n1. The EXACT company name (not the job board name like 'Ashbyhq' or 'Lever')\n2. The actual responsibilities listed (not as placeholders)\n3. The specific skills required\n\nIf you can't find something, simply omit it rather than using placeholders."
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

def query_perplexity(query):
    """Query Perplexity API with a simple question and return the response"""
    try:
        if PERPLEXITY_API_KEY:
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-small",
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.0
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "Unknown")
            else:
                return "Unknown"
        else:
            return "Unknown"
    except:
        return "Unknown" 