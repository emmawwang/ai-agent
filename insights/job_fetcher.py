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
    """Fetch job description from a URL using Perplexity API"""
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key not found. Check your .env file.")
    
    print(f"Fetching job description from URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create query to extract job description
    # Perplexity API payload based on official format
    payload = {
        "model": "sonar",  # Using the base sonar model which has web browsing capability
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful job description extractor. Your task is to visit the provided URL and extract the complete job description in a structured format. Include all sections like responsibilities, requirements, qualifications, and benefits. Format as plain text maintaining the structure of the posting. Don't include your own comments or analysis."
            },
            {
                "role": "user",
                "content": f"Please visit this job posting URL and extract the complete job description: {url}"
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
                return job_description
            else:
                raise ValueError("Empty response received from Perplexity API")
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        print(f"Error fetching job from URL: {e}")
        raise 