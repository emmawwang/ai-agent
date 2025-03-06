import os
from mistralai import Mistral
import discord
import json
import datetime
from datetime import datetime, timedelta

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """
You are pingpal, a helpful assistant designed to help users track their job applications and interviews.
You can track job applications, update their status, and remind users about follow-up emails.

Available commands:
1. !process company_name status [date] - Add or update a job application
   - Status options: applied, oa (online assessment), phone, superday, offer, rejected
   - Date format: YYYY-MM-DD (optional, defaults to today)
   - Example: !process Google applied 2025-02-15

2. !list - List all your job applications
3. !upcoming - Show upcoming interviews
4. !followups - Show applications that need follow-up emails
5. !delete - Delete an existing job application

Respond in a friendly, encouraging tone. Always confirm actions you've taken.
"""

# Define the path for our database file
DB_FILE = "job_tracker_db.json"

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.load_database()

    def load_database(self):
        """Load the database from file or create it if it doesn't exist"""
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {}
            self.save_database()

    def save_database(self):
        """Save the database to file"""
        with open(DB_FILE, 'w') as f:
            json.dump(self.db, f, indent=4)

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
        
        company = args[0]
        role = args[1]
        # Check if role is accidentally a status
        valid_statuses = ["applied", "oa", "phone", "superday", "offer", "rejected"]
        if role.lower() in valid_statuses:
            return "Invalid role name. Please provide a proper role name (i.e. SWE, APM etc). "
        status = args[2].lower()
        
        # Validate status
        valid_statuses = ["applied", "oa", "phone", "superday", "offer", "rejected"]
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
        
        # Initialize user in db if not exists
        if user_id not in self.db:
            self.db[user_id] = {"username": username, "jobs": {}}
        
        # Initialize company in db if not exists
        if company not in self.db[user_id]["jobs"]:
            self.db[user_id]["jobs"][company] = {}
        if role not in self.db[user_id]["jobs"][company]:
            self.db[user_id]["jobs"][company][role] = {}
        
        # Update status/date
        self.db[user_id]["jobs"][company][role][status] = date
        
        # Save changes
        self.save_database()
        
        return f"Updated {company} ({role}) to status '{status}' on {date}."

    def list_jobs(self, user_id):
        """List all job applications for a user"""
        if user_id not in self.db or not self.db[user_id]["jobs"]:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        jobs = self.db[user_id]["jobs"]
        response = "Your job applications:\n\n"
        
        for company, roles_dict in jobs.items():
            response += f"**{company}**\n"
            for role, statuses in roles_dict.items():
                response += f"  - {role}:\n"
                for status, date in sorted(statuses.items()):
                    response += f"      • {status.capitalize()}: {date}\n"
            response += "\n"
        
        return response

    def show_upcoming(self, user_id):
        """Show upcoming interviews based on the latest status of each application"""
        if user_id not in self.db or not self.db[user_id]["jobs"]:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        jobs = self.db[user_id]["jobs"]
        response = "Upcoming interviews:\n\n"
        
        today = datetime.now().date()
        found_upcoming = False
        
        for company, roles_dict in jobs.items():
        # Iterate through each role under this company
            for role, statuses in roles_dict.items():
                # If the user already has "rejected" or "offer" for this role, skip it
                if "rejected" in statuses or "offer" in statuses:
                    continue
                
                # Pull out just the relevant interview stages for this role
                interview_stages = {s: d for s, d in statuses.items() 
                                    if s in ["oa", "phone", "superday"]}
                
                # For each interview stage, check if it's coming up
                for stage, date_str in interview_stages.items():
                    try:
                        interview_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        # If the date in the database is in the wrong format, skip gracefully
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
        if user_id not in self.db or not self.db[user_id]["jobs"]:
            return "You haven't tracked any job applications yet. Use !process to add one."
        
        jobs = self.db[user_id]["jobs"]
        response = "Applications that may need follow-ups:\n\n"
        
        today = datetime.now().date()
        found_followups = False
        
        for company, statuses in jobs.items():
            # Skip if rejected or offered
            if "rejected" in statuses or "offer" in statuses:
                continue
                
            # Get the latest status
            latest_status = None
            latest_date = None
            
            for status, date_str in statuses.items():
                status_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if latest_date is None or status_date > latest_date:
                    latest_status = status
                    latest_date = status_date
            
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
                    response += f"**{company}** - {days_since} days since {latest_status} on {latest_date}. Consider sending a follow-up email.\n"
                    found_followups = True
        
        if not found_followups:
            return "No applications currently need follow-ups. You're on top of things!"
            
        return response
    

    def delete_job(self, args, user_id, username):
        """
        Delete a specific job application from the user's list.
        Usage: !delete company_name
        """
        if len(args) < 2:
            return "Please provide both the company name and the role. Format: !delete <company> <role>"
        
        company = args[0]
        role = args[1]

        # Check if the user has any jobs tracked
        if user_id not in self.db or not self.db[user_id]["jobs"]:
            return "You don't have any job applications to delete."

        # Check if the given company and role are in the user's tracked jobs
        if (
            company not in self.db[user_id]["jobs"] or 
            role not in self.db[user_id]["jobs"][company]
        ):
            return f"No record found for '{company}' with role '{role}'."

        # Delete the role from the user's jobs
        del self.db[user_id]["jobs"][company][role]

        # If that company now has no more roles, delete the company entry too
        if not self.db[user_id]["jobs"][company]:
            del self.db[user_id]["jobs"][company]

        self.save_database()

        return f"Successfully deleted '{company}' ({role})."


    async def run(self, message: discord.Message):
        user_id = str(message.author.id)
        username = message.author.name
        
        # Check if message is a command
        if message.content.startswith("!"):
            return self.process_command(message.content, user_id, username)
        
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
