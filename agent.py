import os
from mistralai import Mistral
import discord
import json
import datetime
from datetime import datetime, timedelta
from supabase import create_client

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """
You are pingpal, a helpful assistant designed to help users track their job applications and interviews.
You can track job applications, update their status, and remind users about follow-up emails.

Available commands:
1. !process company_name role status [date] - Add or update a job application
   - Status options: applied, oa (online assessment), phone, superday, offer, rejected
   - Date format: YYYY-MM-DD (optional, defaults to today)
   - Example: !process Google SWE-intern applied 2025-02-15

2. !list - List all your job applications
3. !upcoming - Show upcoming interviews
4. !followups - Show applications that need follow-up emails
5. !delete - Delete an existing job application

Respond in a friendly, encouraging tone. Always confirm actions you've taken.
"""

class MistralAgent:
    def __init__(self):
        # Initialize Mistral client
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Initialize Supabase client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Cache for frequently accessed data
        self.user_data_cache = {}

    def get_user_data(self, user_id):
        """Get user data from cache or Supabase"""
        # Check cache first
        if user_id in self.user_data_cache:
            return self.user_data_cache[user_id]
        
        # Query Supabase for user data
        response = self.supabase.table('job_tracker_data').select('*').eq('user_id', user_id).execute()
        
        # If user exists in database
        if response.data and len(response.data) > 0:
            user_record = response.data[0]
            # Cache the data
            self.user_data_cache[user_id] = user_record
            return user_record
        
        return None

    def clear_cache(self, user_id):
        """Clear cached data for a user"""
        if user_id in self.user_data_cache:
            del self.user_data_cache[user_id]

    def process_command(self, message_content, user_id, username):
        """Process commands starting with !"""
        parts = message_content.split()
        command = parts[0].lower()

        if command == "!process":
            return self.process_job(parts[1:], user_id, username)
        elif command == "!list":
            return self.list_jobs(user_id)
        elif command == "!upcoming":
            return self.show_upcoming(user_id)
        elif command == "!followups":
            return self.show_followups(user_id)
        elif command == "!delete":
            return self.delete_job(parts[1:], user_id, username)
        else:
            return f"Unknown command: {command}. Try !process, !list, !upcoming, !followups, or !delete."

    def process_job(self, args, user_id, username):
        """Add or update a job application"""
        if len(args) < 3:
            return (
                "Please provide company name, role, and status. "
                "Format: !process <company> <role> <status> [date]"
            )
        
        company = args[0].lower()
        role = args[1].lower()
        # Check if role is accidentally a status
        valid_statuses = ["applied", "oa", "phone", "superday", "offer", "rejected"]
        if role.lower() in valid_statuses:
            return "Invalid role name. Please provide a proper role name (i.e. SWE, APM etc). "
        status = args[2].lower()
        
        # Validate status
        if status not in valid_statuses:
            return f"Invalid status. Please use one of: {', '.join(valid_statuses)}"
        
        # Parse date if provided; otherwise, use today
        date = datetime.now().strftime("%Y-%m-%d")
        if len(args) >= 4:
            try:
                datetime.strptime(args[3], "%Y-%m-%d")
                date = args[3]
            except ValueError:
                return "Invalid date format. Please use YYYY-MM-DD."
        
        # Get existing user record or create new one
        user_record = self.get_user_data(user_id)
        current_time = datetime.now().isoformat()
        
        if user_record:
            # User exists, update their data
            user_data = user_record['data'] if user_record['data'] else {}
            
            # Ensure structure has username and jobs
            if 'username' not in user_data:
                user_data['username'] = username
            if 'jobs' not in user_data:
                user_data['jobs'] = {}
                
            job_data = user_data['jobs']
            
            # Initialize company in data if not exists
            if company not in job_data:
                job_data[company] = {}
            
            # Initialize role in company if not exists
            if role not in job_data[company]:
                job_data[company][role] = {}
            
            # Update status/date
            job_data[company][role][status] = date
            
            # Update the record in Supabase
            self.supabase.table('job_tracker_data').update({
                'data': user_data,
                'updated_at': current_time
            }).eq('id', user_record['id']).execute()
            
        else:
            # Create new user record
            job_data = {
                'username': username,
                'jobs': {
                    company: {
                        role: {
                            status: date
                        }
                    }
                }
            }
            
            self.supabase.table('job_tracker_data').insert({
                'user_id': user_id,
                'data': job_data,
                'created_at': current_time,
                'updated_at': current_time
            }).execute()
        
        # Clear the cache
        self.clear_cache(user_id)
        
        return f"Updated {company} ({role}) to status '{status}' on {date}."

    def list_jobs(self, user_id):
        """List all job applications for a user"""
        user_record = self.get_user_data(user_id)
        
        if not user_record or not user_record['data']:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        user_data = user_record['data']
        
        # Handle both formats (nested with 'jobs' or direct)
        if 'jobs' in user_data:
            jobs = user_data['jobs']
        else:
            jobs = user_data  # For backwards compatibility
            
        if not jobs:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        username = user_data.get("username", "Your")
        response = f"**{username}'s job applications:**\n\n"
        # Define a natural progression for statuses instead of alphabetical
        STATUS_ORDER = ["applied", "oa", "phone", "superday", "offer", "rejected"]
        CAPS_ROLES = ["apm", "swe", "ib", "mle", "pm", "ml", "ai", "hr", "it", "pe", "da", "qa", "ux", "ui", "cto", "cfo", "ceo", "vp", "tpm", "pmo", "cs", "sa", "de", "dba", "ds", "re", "am", "bd", "sm", "po"]
        # Build response from the JSONB data
        for company, roles in jobs.items():
            response += f"**{company.capitalize()}**"
            for role, statuses in roles.items():
                response += f"\n    • {role.upper() if role.lower() in CAPS_ROLES else role}:\n"
                for status in STATUS_ORDER:
                    if status in statuses:
                        date = statuses[status]
                        response += f"      • {status.capitalize()}: {date}\n"
            response += "\n"

        return response


    def show_upcoming(self, user_id):
        """Show upcoming interviews based on the latest status of each application"""
        user_record = self.get_user_data(user_id)
        
        if not user_record or not user_record['data']:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        user_data = user_record['data']
        
        # Handle both formats (nested with 'jobs' or direct)
        if 'jobs' in user_data:
            jobs = user_data['jobs']
        else:
            jobs = user_data  # For backwards compatibility
            
        if not jobs:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        response = "Upcoming interviews:\n\n"
        
        today = datetime.now().date()
        found_upcoming = False
        
        for company, roles in jobs.items():
            for role, statuses in roles.items():
                # If the user already has "rejected" or "offer" for this role, skip it
                if "rejected" in statuses or "offer" in statuses:
                    continue
                
                # Pull out just the relevant interview stages
                interview_stages = {s: d for s, d in statuses.items() 
                                    if s in ["oa", "phone", "superday"]}
                
                # For each interview stage, check if it's coming up
                for stage, date_str in interview_stages.items():
                    try:
                        interview_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        # If the date is in the wrong format, skip gracefully
                        continue
                    
                    # If interview is today or in the future
                    if interview_date >= today:
                        days_until = (interview_date - today).days
                        if days_until == 0:
                            time_info = "Today"
                        elif days_until == 1:
                            time_info = "Tomorrow"
                        else:
                            time_info = f"In {days_until} days"
                            
                        response += (
                            f"**{company}** ({role}) - {stage.capitalize()} interview "
                            f"on {date_str} ({time_info})\n"
                        )
                        found_upcoming = True
        
        if not found_upcoming:
            return "You don't have any upcoming interviews. Keep applying!"
        
        return response

    def show_followups(self, user_id):
        """Show applications that may need follow-up emails"""
        user_record = self.get_user_data(user_id)
        
        if not user_record or not user_record['data']:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        user_data = user_record['data']
        
        # Handle both formats (nested with 'jobs' or direct)
        if 'jobs' in user_data:
            jobs = user_data['jobs']
        else:
            jobs = user_data  # For backwards compatibility
            
        if not jobs:
            return "You haven't tracked any job applications yet. Use !process to add one."
            
        response = "Applications that may need follow-ups:\n\n"
        
        today = datetime.now().date()
        found_followups = False
        
        for company, roles in jobs.items():
            for role, statuses in roles.items():
                # Skip if rejected or offered
                if "rejected" in statuses or "offer" in statuses:
                    continue
                    
                # Get the latest status
                latest_status = None
                latest_date = None
                
                for status, date_str in statuses.items():
                    try:
                        status_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        if latest_date is None or status_date > latest_date:
                            latest_status = status
                            latest_date = status_date
                    except ValueError:
                        # Skip invalid dates
                        continue
                
                # Define follow-up thresholds based on status
                thresholds = {
                    "applied": 14,  # 2 weeks after application
                    "oa": 7,        # 1 week after online assessment
                    "phone": 7,     # 1 week after phone interview
                    "superday": 7   # 1 week after superday
                }
                
                # If the latest action was a while ago, suggest follow-up
                if latest_status in thresholds:
                    days_since = (today - latest_date).days
                    threshold = thresholds[latest_status]
                    
                    if days_since >= threshold:
                        response += f"**{company}** ({role}) - {days_since} days since {latest_status} on {latest_date}. Consider sending a follow-up email.\n"
                        found_followups = True
        
        if not found_followups:
            return "No applications currently need follow-ups. You're on top of things!"
            
        return response
    
    def delete_job(self, args, user_id, username):
        """
        Delete a specific job application from the user's list.
        Usage: !delete company_name role
        """
        if len(args) < 2:
            return "Please provide both the company name and the role. Format: !delete <company> <role>"
        
        company = args[0].lower()
        role = args[1].lower()

        # Get existing user record
        user_record = self.get_user_data(user_id)
        
        # Check if the user has any jobs tracked
        if not user_record or not user_record['data']:
            return "You don't have any job applications to delete."
        
        user_data = user_record['data']
        
        # Handle both formats (nested with 'jobs' or direct)
        if 'jobs' in user_data:
            jobs = user_data['jobs']
        else:
            jobs = user_data  # For backwards compatibility
            
        if not jobs:
            return "You don't have any job applications to delete."
        
        # Check if the given company and role exist
        if (
            company not in jobs or 
            role not in jobs[company]
        ):
            return f"No record found for '{company}' with role '{role}'."
        
        # Delete the role from the user's jobs
        del jobs[company][role]
        
        # If that company now has no more roles, delete the company entry too
        if not jobs[company]:
            del jobs[company]
        
        # Update record in Supabase
        self.supabase.table('job_tracker_data').update({
            'data': user_data,
            'updated_at': datetime.now().isoformat()
        }).eq('id', user_record['id']).execute()
        
        # Clear the cache
        self.clear_cache(user_id)
        
        return f"Successfully deleted '{company}' ({role})."

    async def run(self, message: discord.Message):
        user_id = str(message.author.id)
        username = message.author.name
        
        # Check if message is a command
        if message.content.startswith("!"):
            result = self.process_command(message.content, user_id, username)
            # If result is None, it means this command should be handled by Discord, not the agent
            if result is None:
                return ""
            return result
        
        # If it's not a command, process it as a normal message for Mistral
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.content},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content
