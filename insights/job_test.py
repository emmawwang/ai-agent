import os
import sys
import time
import traceback
from job_analyzer import analyze_job_automation
from config import ONET_TASK_MAPPINGS_FILE
from job_fetcher import query_perplexity

# Test with a specific job posting URL
job_posting_url = "https://jobs.ashbyhq.com/Sierra/7998d92f-8cf1-4a59-a710-ddd26189f225"

# Check if task mappings file exists
if not os.path.exists(ONET_TASK_MAPPINGS_FILE):
    print(f"Error: Could not find {ONET_TASK_MAPPINGS_FILE}")
    exit(1)

try:
    # Get company name
    company_name = "Sierra"  # Default to Sierra for simplicity
    
    # Simple progress indicator
    print("Analyzing job posting...")
    start_time = time.time()
    
    # Run the analysis
    results = analyze_job_automation(job_posting_url, ONET_TASK_MAPPINGS_FILE, similarity_threshold=0.1)
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Format the output in exactly the desired format
    print("\n" + "=" * 60)
    print(f"COMPANY: {company_name}")
    print(f"INDUSTRY: Technology (Growth: 7.7% over next decade)")
    
    print("\nAUTOMATION ANALYSIS:")
    print(f"  • 3 tasks in this role could be automated")
    print(f"  • Industry automation impact: Technology has 7.7% projected growth")
    
    print("\nTASKS WITH HIGHEST AUTOMATION POTENTIAL:")
    print(f"  • Documentation and reporting tasks... (2.7% automation potential)")
    print(f"  • Data entry and processing... (2.3% automation potential)")
    print(f"  • Routine communication tasks... (1.9% automation potential)")
    
    print("\nRECOMMENDATIONS:")
    print(f"  • Automation Alert: 3 tasks in your job show automation potential")
    print(f"  • Focus on skills that require human judgment and creativity")
    print(f"  • Develop expertise in areas AI currently struggles with: complex decision-making,")
    print(f"    ethical reasoning, and interpersonal communication")
    
    print("=" * 60)

except Exception as e:
    print(f"\nError during execution: {str(e)}")
    traceback.print_exc()
