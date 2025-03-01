from job_analyzer import analyze_job_automation

# Path to your data
csv_path = "economic_data.csv"

# Test with a specific job description
job_description = """
Software Engineer
Responsibilities:
- Develop and maintain software applications
- Write clean, maintainable code
- Debug issues and perform code reviews
- Collaborate with cross-functional teams
"""

# Analyze
results = analyze_job_automation(job_description, csv_path)

# Print results
print("\nResults:")
print(f"Automation Risk: {results.get('Overall_Automation_Risk', 0):.4f}")
print(f"Career Growth: {results.get('Career_Growth_Potential', 'Unknown')}")
print(f"Industry Stability: {results.get('Industry_Stability', 'Unknown')}")

# Show top matches
if 'Matched_Tasks' in results:
    print("\nTop matching tasks:")
    for i, task in enumerate(results['Matched_Tasks'][:3]):
        print(f"{i+1}. '{task['job_task'][:50]}...'")
        print(f"   Matched to: '{task['matched_task'][:50]}...'")
        print(f"   Automation: {task['automation_pct']*100:.2f}%") 