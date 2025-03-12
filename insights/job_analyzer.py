import pandas as pd
import numpy as np
import os
import requests
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from insights.config import BLS_EMPLOYMENT_FILE, ONET_TASK_MAPPINGS_FILE
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

class JobAnalyzer:
    def __init__(self, task_mappings_file=ONET_TASK_MAPPINGS_FILE):
        self.task_data = pd.read_csv(task_mappings_file)
        self.model = model
        
    def analyze_job(self, job_url):
        """Main analysis pipeline"""
        try:
            print("\n=== Starting Job Analysis ===")
            
            # 1. Get job description
            job_info = self._fetch_job_info(job_url)
            print(f"Job info keys: {job_info.keys() if isinstance(job_info, dict) else 'Not a dict'}")
            
            if 'error' in job_info:
                return job_info
                
            if 'industry' not in job_info:
                print("⚠️ No industry found in job info")
                job_info['industry'] = 'Technology'  # Default fallback
            
            # 2. Extract tasks and analyze automation
            tasks = self._extract_tasks(job_info['description'])
            print(f"Extracted {len(tasks)} tasks")
            
            automatable_tasks = self._analyze_automation(tasks)
            print(f"Analyzed {len(automatable_tasks)} tasks for automation")
            
            # 3. Compile results
            results = {
                'company': job_info['company'],
                'industry': job_info['industry'],
                'tasks': tasks,
                'automation_analysis': automatable_tasks
            }
            
            print("Analysis complete with keys:", results.keys())
            return results
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {'error': f"Analysis failed: {str(e)}"}
            
    def _fetch_job_info(self, url):
        """Fetch job info using Perplexity"""
        logger.info(f"Attempting to fetch job info from URL: {url}")
        
        try:
            # First try to fetch the page content directly
            logger.info("Attempting direct URL access")
            response = requests.get(url)
            response.raise_for_status()
            page_content = response.text
            logger.info("Successfully fetched page content")
            
            # Define the prompt with the actual page content
            prompt = f"""
            Analyze this job posting content:
            {page_content[:4000]}  # First 4000 chars to stay within token limits
            
            Extract EXACT tasks and responsibilities. Do not summarize.
            
            Format:
            Company: [exact company name]
            Industry: [closest match from categories below]
            Tasks:
            - [COPY-PASTE exact task from the content above]
            - [COPY-PASTE exact task from the content above]
            - [COPY-PASTE exact task from the content above]
            
            Available industry categories:
            - Computer and mathematical
            - Healthcare practitioners and technical
            [... rest of categories ...]
            """
            
            # Query Perplexity with the content
            response = self._query_perplexity(prompt)
            logger.info(f"Raw Perplexity response: {response[:500]}")
            
            # Parse the response
            result = {
                'description': response,
                'company': self._extract_field(response, 'Company', 'Mercor'),
                'industry': self._extract_field(response, 'Industry', 'Computer and mathematical'),
                'tasks': self._extract_tasks(response)
            }
            
            # Validate we got tasks
            if not result['tasks']:
                logger.warning("No tasks found in Perplexity response, trying direct extraction")
                # Try to extract tasks directly from page content
                direct_tasks = self._extract_tasks_from_html(page_content)
                if direct_tasks:
                    result['tasks'] = direct_tasks
                    logger.info(f"Found {len(direct_tasks)} tasks directly from HTML")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to access URL directly: {str(e)}")
            return {'error': f"Could not access job posting. Error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error in _fetch_job_info: {str(e)}", exc_info=True)
            return {'error': f"Failed to fetch job info: {str(e)}"}

    def _extract_tasks_from_html(self, html_content):
        """Extract tasks directly from HTML content"""
        try:
            # Common patterns for task sections in job postings
            patterns = [
                r'responsibilities:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'requirements:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'what you\'ll do:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'job duties:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'essential functions:?\s*((?:[-•*]\s*[^\n]+\n*)+)'
            ]
            
            tasks = []
            for pattern in patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Extract bullet points
                    bullet_points = re.findall(r'[-•*]\s*([^\n]+)', match)
                    tasks.extend([t.strip() for t in bullet_points if len(t.strip()) > 10])
            
            return tasks[:5]  # Return top 5 most relevant tasks
            
        except Exception as e:
            logger.error(f"Error extracting tasks from HTML: {str(e)}")
            return []
        
    def _extract_tasks(self, description):
        """Extract tasks from Perplexity response"""
        try:
            logger.info("Attempting to extract tasks from description")
            logger.info(f"Description snippet: {description[:200]}...")
            
            # First try to find a Tasks section
            tasks_section = re.search(r'Tasks:?\s*((?:[-•*]\s*[^\n]+\n*)+)', description, re.IGNORECASE | re.DOTALL)
            
            if tasks_section:
                logger.info("Found Tasks section")
                # Extract bullet points from the Tasks section
                tasks = re.findall(r'[-•*]\s*([^\n]+)', tasks_section.group(1))
                # Clean up tasks and remove template placeholders
                cleaned_tasks = [t.strip() for t in tasks if t.strip() and not t.strip().startswith('[') and len(t.strip()) > 10]
                logger.info(f"Extracted {len(cleaned_tasks)} tasks")
                return cleaned_tasks
            else:
                logger.warning("No Tasks section found, trying alternative extraction")
                # Try to find bullet points anywhere in the text
                all_bullets = re.findall(r'[-•*]\s*([^\n]+)', description)
                potential_tasks = [t.strip() for t in all_bullets if t.strip() and len(t.strip()) > 10 
                                 and not t.strip().startswith('[') 
                                 and any(keyword in t.lower() for keyword in ['develop', 'create', 'manage', 'design', 'implement', 'analyze', 'work'])]
                
                if potential_tasks:
                    logger.info(f"Found {len(potential_tasks)} potential tasks using alternative method")
                    return potential_tasks[:5]  # Return top 5 most likely tasks
                
                logger.warning("No tasks found in text")
                return []
            
        except Exception as e:
            logger.error(f"Error extracting tasks: {str(e)}")
            return []
        
    def _analyze_automation(self, tasks):
        """Analyze task automation potential using task mappings"""
        try:
            logger.info(f"Analyzing {len(tasks)} tasks for automation potential")
            
            if not tasks:
                logger.warning("No tasks provided for analysis")
                return []
            
            # Get embeddings
            task_embeddings = self.model.encode([t.lower() for t in tasks])
            onet_tasks = self.task_data['task_name'].tolist()
            onet_embeddings = self.model.encode([t.lower() for t in onet_tasks])
            
            # Calculate similarities
            similarities = cosine_similarity(task_embeddings, onet_embeddings)
            
            results = []
            seen_onet_tasks = set()  # Track used ONET tasks to avoid duplicates
            
            for i, task in enumerate(tasks):
                logger.info(f"\nAnalyzing task: {task}")
                best_match = None
                best_score = 0.3  # Minimum similarity threshold
                
                # Find best matching ONET task
                for j, score in enumerate(similarities[i]):
                    if score > best_score and onet_tasks[j] not in seen_onet_tasks:
                        best_score = score
                        best_match = {
                            'task': onet_tasks[j],  # Use ONET task instead of original
                            'match': onet_tasks[j],
                            'score': float(score),
                            'automation_potential': float(self.task_data.iloc[j]['pct'])
                        }
                
                if best_match:
                    results.append(best_match)
                    seen_onet_tasks.add(best_match['task'])
                    logger.info(f"Matched to ONET task: {best_match['task']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in task analysis: {str(e)}", exc_info=True)
            return []
            
    def _query_perplexity(self, prompt):
        """Query Perplexity API"""
        logger.info("Querying Perplexity API")
        try:
            url = "https://api.perplexity.ai/chat/completions"
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise. Extract exact information from job postings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": None,
                "return_images": False,
                "return_related_questions": False,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1,
                "response_format": None
            }
            
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            logger.info("Sending request to Perplexity API")
            response = requests.post(url, json=payload, headers=headers)
            
            logger.info(f"Perplexity API status code: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                logger.error("No choices in Perplexity response")
                raise Exception("No choices in response")
            
            content = result['choices'][0]['message']['content']
            logger.info(f"Received response: {content[:200]}...")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing Perplexity response: {str(e)}")
            raise
        
    def _extract_field(self, text, field, default=None):
        """Extract a field from text"""
        pattern = rf'{field}:?\s*([^\n]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def analyze_job_description(self, job_description):
        """Analyze a job description text directly
        
        Parameters:
        job_description (str): The text of the job description
        
        Returns:
        dict: Results of the job analysis
        """
        try:
            print("\n=== Starting Job Description Analysis ===")
            
            # 1. Extract job info from text instead of fetching from URL
            job_info = self._parse_job_description(job_description)
            
            if 'error' in job_info:
                return job_info
                
            if 'industry' not in job_info:
                print("⚠️ No industry found in job info")
                job_info['industry'] = 'Technology'  # Default fallback
            
            # 2. Extract tasks and analyze automation
            tasks = self._extract_tasks(job_description)
            print(f"Extracted {len(tasks)} tasks")
            
            automatable_tasks = self._analyze_automation(tasks)
            print(f"Analyzed {len(automatable_tasks)} tasks for automation")
            
            # 3. Compile results
            results = {
                'company': job_info.get('company', 'Not specified'),
                'industry': job_info.get('industry', 'Technology'),
                'tasks': tasks,
                'automation_analysis': automatable_tasks
            }
            
            print("Analysis complete with keys:", results.keys())
            return results
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {'error': f"Analysis failed: {str(e)}"}
    

    def _parse_job_description(self, description):
        """Parse job description text to extract relevant information
        
        Parameters:
        description (str): The text of the job description
        
        Returns:
        dict: Extracted job information
        """
        try:
            logger.info("Parsing job description text")
            
            # Use Perplexity to analyze the job description
            prompt = f"""
            Analyze this job description:
            {description[:4000]}  # First 4000 chars to stay within token limits
            
            Extract EXACT tasks and responsibilities. Do not summarize.
            
            Format:
            Company: [company name if mentioned, or "Not specified"]
            Industry: [closest match from categories below]
            Tasks:
            - [COPY-PASTE exact task from the content above]
            - [COPY-PASTE exact task from the content above]
            - [COPY-PASTE exact task from the content above]
            
            Available industry categories:
            - Computer and mathematical
            - Healthcare practitioners and technical
            - Management
            - Business and financial operations
            - Sales and related
            - Education, training, and library
            - Office and administrative support
            - Architecture and engineering
            - Arts, design, entertainment, sports, and media
            - Life, physical, and social science
            - Legal
            - Community and social service
            - Construction and extraction
            - Installation, maintenance, and repair
            - Production
            - Transportation and material moving
            - Farming, fishing, and forestry
            - Protective service
            - Personal care and service
            - Food preparation and serving related
            - Building and grounds cleaning and maintenance
            """
            
            # Query Perplexity with the content
            response = self._query_perplexity(prompt)
            logger.info(f"Raw Perplexity response: {response[:500]}")
            
            # Parse the response
            result = {
                'description': description,  # Store original description
                'company': self._extract_field(response, 'Company', 'Not specified'),
                'industry': self._extract_field(response, 'Industry', 'Computer and mathematical'),
                'tasks': self._extract_tasks(response)
            }
            
            # Validate we got tasks
            if not result['tasks']:
                logger.warning("No tasks found in Perplexity response, trying direct extraction")
                # Try to extract tasks directly from the description
                direct_tasks = self._extract_tasks_from_text(description)
                if direct_tasks:
                    result['tasks'] = direct_tasks
                    logger.info(f"Found {len(direct_tasks)} tasks directly from text")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing job description: {str(e)}", exc_info=True)
            return {'error': f"Failed to parse job description: {str(e)}"}

    def _extract_tasks_from_text(self, text):
        """Extract tasks directly from plain text description"""
        try:
            # Common patterns for task sections in job descriptions
            patterns = [
                r'responsibilities:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'requirements:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'what you\'ll do:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'job duties:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'essential functions:?\s*((?:[-•*]\s*[^\n]+\n*)+)',
                r'key responsibilities:?\s*((?:[-•*]\s*[^\n]+\n*)+)'
            ]
            
            tasks = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Extract bullet points
                    bullet_points = re.findall(r'[-•*]\s*([^\n]+)', match)
                    tasks.extend([t.strip() for t in bullet_points if len(t.strip()) > 10])
            
            # If no bullet points found, look for sentences that might be tasks
            if not tasks:
                sentences = re.split(r'[.!?]\s+', text)
                potential_tasks = [s.strip() for s in sentences if len(s.strip()) > 10 
                                and any(keyword in s.lower() for keyword in 
                                    ['develop', 'create', 'manage', 'design', 'implement', 
                                    'analyze', 'work', 'maintain', 'ensure', 'coordinate'])]
                tasks = potential_tasks[:5]  # Take top 5 most likely task sentences
            
            return tasks[:5]  # Return top 5 most relevant tasks
            
        except Exception as e:
            logger.error(f"Error extracting tasks from text: {str(e)}")
            return []
    
def analyze_job_automation(job_input, is_url, task_mappings_file=ONET_TASK_MAPPINGS_FILE):
    """Wrapper function for job analysis
    
    Parameters:
    job_input (str): Either a URL to a job posting or the actual job description text
    is_url (bool): Flag indicating whether job_input is a URL or a description
    task_mappings_file (str): Path to the task mappings file
    
    Returns:
    dict: Results of the job analysis
    """
    logger.info(f"Starting job analysis for {'URL' if is_url else 'text description'}")
    
    try:
        analyzer = JobAnalyzer(task_mappings_file)
        
        if is_url:
            results = analyzer.analyze_job(job_input)
        else:
            results = analyzer.analyze_job_description(job_input)
        
        if 'error' in results:
            logger.warning(f"Analysis returned error: {results['error']}")
        else:
            logger.info("Analysis completed successfully")
            if 'industry' in results:
                logger.info(f"Industry detected: {results['industry']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_job_automation: {str(e)}", exc_info=True)
        return {'error': f"Analysis failed: {str(e)}"}
    
def get_growth_thresholds(bls_data):
    """Calculate growth thresholds from BLS data"""
    try:
        print("\nColumns in BLS data:", bls_data.columns.tolist())
        # Use the correct column name from the CSV
        growth_rates = bls_data['Annualized Employment Change (%)'].astype(float).values
        high_threshold = np.percentile(growth_rates, 75)  # Top 25% is high growth
        low_threshold = np.percentile(growth_rates, 25)   # Bottom 25% is low growth
        return high_threshold, low_threshold
    except Exception as e:
        print(f"Error in get_growth_thresholds: {str(e)}")
        return 0.7, 0.3  # Default thresholds

def get_career_recommendations(matched_category, annual_growth, automation_level, tasks):
    """Get dynamic career recommendations using Mistral"""
    try:
        prompt = f"""
        As a career advisor, provide specific recommendations for someone in the {matched_category} field, focusing on AI automation adaptation.

        Context:
        - Field: {matched_category}
        - Annual Job Growth: {annual_growth:+.1f}%
        - AI Automation Level: {automation_level}
        - Current Tasks: {', '.join(tasks[:3])}

        Provide very specific recommendations in exactly this format, focusing on staying relevant in the age of AI:

        Market Analysis:
        • Current Trend: [specific trend in {matched_category} field, mention growth rate]
        • Key Opportunity: [specific opportunity based on AI transformation]

        Critical Skills to Develop:
        • Technical: [specific technical skill that complements AI, not replaced by it]
        • Human-Centric: [specific interpersonal/strategic skill that AI can't replicate]
        • Emerging: [specific emerging technology or methodology in {matched_category}]

        Concrete Action Plan:
        • Immediate Step: [specific certification/course with actual name]
        • 6-Month Goal: [specific project or skill milestone]
        • Career Pivot: [specific way to position oneself as AI-collaboration expert]

        Remember to be very specific - mention actual technologies, certifications, or methodologies relevant to {matched_category}.
        """
        
        response = query_mistral(prompt)
        return response.strip()
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return None

def get_skill_suggestions(tasks, automation_level, matched_category):
    """Generate dynamic skill suggestions based on tasks and automation"""
    try:
        prompt = f"""
        Given these job tasks and context, suggest 3 specific skills to develop.
        Focus on skills that complement AI and automation.
        
        Tasks: {tasks}
        Field: {matched_category}
        AI Automation Level: {automation_level}
        
        Format your response exactly like this:
        • [Skill Category]: [Specific skill or technology]
        • [Skill Category]: [Specific skill or technology]
        • [Skill Category]: [Specific skill or technology]
        
        Be very specific with actual technologies, methodologies, or frameworks.
        """
        
        response = query_mistral(prompt)
        if response:
            return response.strip().split('\n')
        return None
        
    except Exception as e:
        print(f"Error getting skill suggestions: {e}")
        return None

def get_task_based_recommendations(tasks, matched_category, annual_growth, automation_analyses):
    """Generate concise recommendations using Mistral"""
    try:
        # Categorize tasks by automation risk
        high_risk_tasks = []
        medium_risk_tasks = []
        
        for analysis in automation_analyses:
            task = analysis['task']
            if analysis['automation_potential'] > 0.03:
                high_risk_tasks.append(task)
            elif analysis['automation_potential'] > 0.005:
                medium_risk_tasks.append(task)

        # Build concise Mistral prompt
        prompt = f"""
        Given these tasks with automation risk:

        High Risk: {', '.join(high_risk_tasks) if high_risk_tasks else 'None'}
        Medium Risk: {', '.join(medium_risk_tasks) if medium_risk_tasks else 'None'}
        Industry: {matched_category}

        Provide 1-2 specific skills that complement (not compete with) these tasks.
        Be extremely concise - one sentence per skill, mentioning specific technologies.
        
        Format exactly like this:
        [Only include sections if tasks exist]

        📚 Skills to Complement Automation:

        ⚠️ High Risk Tasks → [specific skill/technology]: One sentence about how it complements AI
        💡 Medium Risk Tasks → [specific skill/technology]: One sentence about enhancement
        """

        recommendations = query_mistral(prompt)
        if recommendations:
            # Add header if not present in Mistral's response
            if "📚 Skills to Complement Automation:" not in recommendations:
                recommendations = "\n📚 Skills to Complement Automation:\n" + recommendations
            return recommendations.split('\n')
        
        return ["\n⚠️ Error: Could not generate recommendations"]
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return ["\n⚠️ Error: Could not generate skill recommendations"]

def format_results_for_discord(results):
    """Format results for job seekers"""
    if 'error' in results:
        if "Could not access job posting" in results['error']:
            return (
                "```\n"
                "❌ Unable to analyze this job posting\n\n"
                "Perplexity cannot properly scrape this website. For job analysis:\n"
                "• Try a different job board (LinkedIn, Indeed, etc.)\n"
                "• Use a public job posting URL\n"
                "• Copy-paste the job description directly\n"
                "```"
            )
        return f"❌ Error: {results['error']}"
    
    try:
        # Get industry growth rate first
        industry = results['industry']
        annual_growth, matched_category = get_industry_growth_rate(industry)
        
        output = [
            "```",
            "🎯 Job Analysis for Career Planning",
            "=" * 40,
            f"\n📊 Company: {results.get('company', 'Unknown')}",
            f"🔮 Category: {matched_category}",
            f"   Job Growth: {annual_growth:+.1f}% (2023-33 projection)",
        ]
        
        if not results.get('automation_analysis'):
            output.extend([
                "\n⚠️ Task Analysis Unavailable",
                "Could not extract specific tasks from this job posting.",
                "\nFor detailed analysis, please try:",
                "• A different job board (LinkedIn, Indeed)",
                "• Copy-paste the job description directly"
            ])
        else:
            # Sort tasks by automation potential and get top 3
            sorted_analyses = sorted(
                results['automation_analysis'], 
                key=lambda x: x['automation_potential'], 
                reverse=True
            )[:3]  # Only take top 3
            
            output.append("\n🤖 Top 3 Tasks by Automation Impact:")
            
            for analysis in sorted_analyses:
                if analysis['automation_potential'] > 0.03:
                    risk_level = "High"
                elif analysis['automation_potential'] > 0.005:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                    
                output.extend([
                    f"\n• {analysis['task']}",
                    f"  AI Impact: {risk_level} automation potential"
                ])
            
            # Get recommendations based on top 3 tasks
            recommendations = get_task_based_recommendations(
                [a['task'] for a in sorted_analyses],
                matched_category,
                annual_growth,
                sorted_analyses
            )
            output.extend(recommendations)
        
        output.append("```")
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return "❌ Error: Unable to analyze job posting. Please try again."

def get_industry_growth_rate(industry):
    """Get BLS annualized employment growth rate using semantic similarity"""
    try:
        print("\n" + "="*50)
        print("🔍 INDUSTRY MATCHING DEBUG LOG")
        print("="*50)
        
        # 1. Load BLS data
        bls_data = pd.read_csv(BLS_EMPLOYMENT_FILE)
        print(f"\nLoaded {len(bls_data)} BLS categories")
        print("Available columns:", bls_data.columns.tolist())
        
        # 2. Clean input industry
        industry_name = industry.lower().replace('industry', '').strip()
        print(f"\nMatching industry: '{industry_name}'")
        
        # 3. Get embeddings for semantic matching
        industry_embedding = model.encode([industry_name])
        # Use correct column name from the CSV
        bls_categories = bls_data['Occupational_Group'].tolist()
        bls_embeddings = model.encode([cat.lower() for cat in bls_categories])
        
        # 4. Calculate similarities and find best match
        similarities = cosine_similarity(industry_embedding, bls_embeddings)[0]
        
        # Show top 3 matches for debugging
        print("\nTop 3 closest matches:")
        top_3_indices = np.argsort(similarities)[-3:][::-1]
        for idx in top_3_indices:
            category = bls_categories[idx]
            score = similarities[idx]
            growth = float(bls_data[bls_data['Occupational_Group'] == category]['Projected_Annualized_Employment_Change(%)'].iloc[0])
            print(f"- {category:<35} (similarity: {score:.3f}, growth: {growth:+.2f}%)")
        
        # Get the best match
        best_match_idx = top_3_indices[0]
        matched_category = bls_categories[best_match_idx]
        growth_rate = float(bls_data[bls_data['Occupational_Group'] == matched_category]['Projected_Annualized_Employment_Change(%)'].iloc[0])
        
        print(f"\nSelected match: {matched_category} ({growth_rate:+.2f}% growth)")
        print("="*50)
        
        return growth_rate, matched_category
        
    except Exception as e:
        print(f"\nError in industry matching: {str(e)}")
        print("Available columns:", bls_data.columns.tolist() if 'bls_data' in locals() else "No data loaded")
        return 1.22, "Computer and mathematical"  # Default to tech industry

def query_mistral(prompt):
    """Query Mistral API for career recommendations"""
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-medium",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a career advisor specializing in AI transformation of industries.
                        Your recommendations should be extremely specific:
                        - Name actual certifications, courses, or technologies
                        - Provide concrete steps, not general advice
                        - Focus on skills that complement AI, not compete with it
                        - Give timeline-based action items
                        - Reference real-world trends and tools
                        Always consider both the growth rate and automation level in your advice."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.4,  # Reduced for more focused responses
                "max_tokens": 500
            }
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"Error querying Mistral: {e}")
        return None

if __name__ == "__main__":
    analyzer = JobAnalyzer()
    url = input("Enter job URL: ")
    results = analyzer.analyze_job(url)
    print(json.dumps(results, indent=2))
