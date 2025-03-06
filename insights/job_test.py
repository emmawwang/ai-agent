import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import traceback
from insights.job_analyzer import analyze_job_automation
from insights.config import ONET_TASK_MAPPINGS_FILE
from insights.job_fetcher import query_perplexity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test with a specific job posting URL
job_posting_url = "https://jobs.ashbyhq.com/Sierra/7998d92f-8cf1-4a59-a710-ddd26189f225"

# Check if task mappings file exists
if not os.path.exists(ONET_TASK_MAPPINGS_FILE):
    print(f"Error: Could not find {ONET_TASK_MAPPINGS_FILE}")
    exit(1)

try:
    # Simple progress indicator
    print("Analyzing job posting...")
    start_time = time.time()
    
    # Run the actual analysis
    results = analyze_job_automation(job_posting_url, ONET_TASK_MAPPINGS_FILE, similarity_threshold=0.1)
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Print the formatted results
    if isinstance(results, dict) and 'error' not in results:
        from insights.job_analyzer import format_results_for_discord
        formatted_output = format_results_for_discord(results)
        print("\n" + formatted_output)
    else:
        print("\nError in analysis:", results.get('error', 'Unknown error occurred'))

except Exception as e:
    print(f"\nError during execution: {str(e)}")
    traceback.print_exc()
