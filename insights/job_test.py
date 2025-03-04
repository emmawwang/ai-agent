import os
import sys
from urllib.parse import urlparse
import traceback
from job_fetcher import fetch_job_from_url

# Test with a specific job posting URL
job_posting_url = "https://jobs.ashbyhq.com/Sierra/7998d92f-8cf1-4a59-a710-ddd26189f225"

print("Running simplified test...")
try:
    # Fetch job description using Perplexity API
    print("Fetching job description...")
    job_description = fetch_job_from_url(job_posting_url)
    
    # Extract company name from job description
    company_name = "Unknown Company"
    
    # Look for common patterns in job descriptions to find company name
    import re
    company_patterns = [
        r'(?:at|join|about|with)\s+([A-Z][A-Za-z0-9\'\-]+(?:\s+[A-Z][A-Za-z0-9\'\-]+){0,2})',  # "at Company Name"
        r'([A-Z][A-Za-z0-9\'\-]+(?:\s+[A-Z][A-Za-z0-9\'\-]+){0,2})\s+is\s+(?:a|an|the)',  # "Company Name is a"
        r'Welcome\s+to\s+([A-Z][A-Za-z0-9\'\-]+(?:\s+[A-Z][A-Za-z0-9\'\-]+){0,2})',  # "Welcome to Company"
        r'About\s+([A-Z][A-Za-z0-9\'\-]+(?:\s+[A-Z][A-Za-z0-9\'\-]+){0,2})'  # "About Company"
    ]
    
    for pattern in company_patterns:
        matches = re.search(pattern, job_description)
        if matches:
            company_name = matches.group(1).strip()
            break
    
    # Fallback to URL parsing if we couldn't extract from job description
    if company_name == "Unknown Company":
        parsed_url = urlparse(job_posting_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if 'ashbyhq.com' in parsed_url.netloc and len(path_parts) > 0:
            company_name = path_parts[0].replace('-', ' ').title()
    
    print(f"\nCompany Name: {company_name}")
    
    # Hardcoded sample output for testing
    print("\nClaude Usage Analysis:")
    print("This job involves tasks that have moderate overlap with Claude's capabilities. The role requires a mix of analytical thinking, communication skills, and domain expertise.")
    
    print(f"\nCareer Growth Potential at {company_name}: Moderate to High")
    print("  • This role provides opportunities for advancement in the AI and technology sector")
    print("  • Projected growth: 15% over the next decade")
    print("  • Key skills: AI development, machine learning, software engineering")
    print("  • Recommendation: Focus on developing specialized technical skills while building domain expertise")

except Exception as e:
    print(f"Error during execution: {str(e)}")
    traceback.print_exc()

print("\nTest completed.")
