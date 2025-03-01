import os
from job_analyzer import analyze_job_automation
from config import ECONOMIC_DATA_FILE, BLS_EMPLOYMENT_FILE, ONET_TASK_MAPPINGS_FILE
import re

# Test with a specific job posting URL
job_posting_url = "https://scale.com/careers/4463083005"

# Handle file paths correctly
if not os.path.exists(ECONOMIC_DATA_FILE):
    print(f"Error: Could not find {ECONOMIC_DATA_FILE}")
    print(f"Current working directory: {os.getcwd()}")
    print("Please make sure the economic_data.csv file exists")
    exit(1)

print("Running analysis...")
results = analyze_job_automation(job_posting_url, ONET_TASK_MAPPINGS_FILE, similarity_threshold=0.1)

# Calculate risk level description
automation_risk = results.get('Overall_Automation_Risk', 0)
if automation_risk < 0.3:
    risk_level = "LOW"
    risk_desc = "This job has low automation risk. Most tasks require human judgment and creativity."
elif automation_risk < 0.6:
    risk_level = "MODERATE" 
    risk_desc = "This job has moderate automation risk. Some tasks could be automated, but human oversight remains important."
else:
    risk_level = "HIGH"
    risk_desc = "This job has high automation risk. Many tasks may be automated in the coming years."

print("\n" + "="*50)
print(f"AUTOMATION RISK ASSESSMENT: {risk_level} ({automation_risk:.4f})")
print("="*50)
print(f"{risk_desc}\n")

# Enhanced but concise career growth output
career = results.get('Career_Growth_Potential', {})
company_info = results.get('Company_Info', {})
company_name = company_info.get('company_name', 'this company') if company_info else 'this company'

print(f"Career Growth Potential: {career.get('level', 'Unknown')}")

# Extract just the key part of the description, keeping it short
description = career.get('description', '')
if len(description) > 100:
    description = description.split('(Source:')[0].strip()  # Remove source citations to be concise
    
print(f"  • {description}")

# Extract just the growth percentage from the outlook
outlook = career.get('outlook', '')
growth_match = re.search(r'grow(?:th|ing).*?(\d+(?:\.\d+)?%)', outlook)
if growth_match:
    growth_pct = growth_match.group(1)
    print(f"  • Projected growth: {growth_pct}")
else:
    # Only keep first sentence or up to 100 chars
    if len(outlook) > 100:
        outlook = outlook.split('.')[0] + '.'
    print(f"  • Outlook: {outlook}")

# Make skill demand more concise by listing just the skills
skill_demand = career.get('skill_demand', '')
if ',' in skill_demand:
    # If it's a list, extract just the list
    skills_list = skill_demand.split(':')[-1].strip()
    print(f"  • Key skills: {skills_list}")
else:
    # Extract first 2-3 skills from a longer text
    skills_extracted = re.findall(r'(?:^|\n)\s*[•-]\s*\*?\*?([^•\n:]+)', skill_demand)
    if skills_extracted:
        skills_concise = "; ".join(s.strip() for s in skills_extracted[:3])
        print(f"  • Key skills: {skills_concise}")
    else:
        print(f"  • {skill_demand[:100]}...")

# Keep the recommendation concise
recommendation = career.get('recommendations', '')
if len(recommendation) > 100:
    recommendation = recommendation.split('.')[0] + '.'
print(f"  • Recommendation: {recommendation}")

if 'Matched_Tasks' in results:
    print("\nAutomation Analysis Breakdown:")
    print("The following tasks from this job were analyzed for automation potential:")
    for i, task in enumerate(results['Matched_Tasks'][:5]):
        risk_emoji = "🟢" if task['automation_pct'] < 0.3 else "🟠" if task['automation_pct'] < 0.6 else "🔴"
        print(f"{i+1}. {risk_emoji} '{task['job_task'][:80]}...'")
        print(f"   → Automation Potential: {task['automation_pct']*100:.1f}% (Confidence: {task['match_score']:.2f})")
        print(f"   → Similar To: '{task['matched_task'][:80]}...'\n")

def test_job_analyzer():
    # ... existing code ...
    results = analyze_job_automation(job_description, ONET_TASK_MAPPINGS_FILE)
    # ... rest of test code ... 