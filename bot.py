import os
import sys
import discord
import logging
from datetime import datetime, timedelta

from discord.ext import commands, tasks
from dotenv import load_dotenv
from agent import MistralAgent
from insights.job_analyzer import analyze_job_automation, format_results_for_discord
from insights.config import ONET_TASK_MAPPINGS_FILE, ECONOMIC_DATA_FILE, BLS_EMPLOYMENT_FILE

PREFIX = "!"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()
load_dotenv(dotenv_path=".env", override=True)

# Create the bot with all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Remove the default help command
bot.remove_command('help')

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

# Help message for the bot
BOT_HELP_MESSAGE = """
**pingpal Bot Commands**

📋 **Track Your Applications**
`!process company_name role status [date]` - Add or update a job application
  • Status options: applied, oa, phone, superday, offer, rejected
  • Date format: YYYY-MM-DD (optional, defaults to today)
  • Example: `!process Google product-design-intern applied 2025-02-15`
`!delete company_name role` - Delete a job application
  • Example: `!delete Google SWE-intern`

📊 **View Your Applications**
`!list` - List all your job applications
`!upcoming` - Show upcoming interviews
`!followups` - Show applications that need follow-up emails
`!delete` - Delete an existing job application

🤖 **Job Analysis**
`!insights [job description or URL]` - Analyze a job posting for automation risk and get detailed job insights
  • Example: `!insights https://example.com/job-posting`
  • Example: `!insights [paste job description here]`

❓ **Help**
`!help` - Show this help message

*The bot will also respond to natural language questions about your job search!*
"""

def simulate_discord():
    """Simulate Discord interaction through terminal"""
    print("\nWelcome to the Job Analysis Bot!")
    print("Enter 'exit' to quit\n")
    
    while True:
        # Get input from user
        print("\nEnter a job posting URL to analyze:")
        url = input().strip()
        
        if url.lower() == 'exit':
            print("Goodbye!")
            break
            
        if not url.startswith('http'):
            print("Please enter a valid URL starting with http:// or https://")
            continue
            
        try:
            print("\nAnalyzing job posting... This may take a moment.")
            
            # Run the analysis
            results = analyze_job_automation(url, ONET_TASK_MAPPINGS_FILE)
            
            # Format and print results
            if isinstance(results, dict) and 'error' not in results:
                formatted_output = format_results_for_discord(results)
                print("\n" + formatted_output)
            else:
                print("\nError in analysis:", results.get('error', 'Unknown error occurred'))
                
        except Exception as e:
            logger.error(f"Error analyzing job: {e}")
            print(f"\nError: {str(e)}")
            print("Please try again with a different URL.")

@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    logger.info(f"{bot.user} has connected to Discord!")
    check_for_reminders.start()  # Start the reminder task


@tasks.loop(hours=24)
async def check_for_reminders():
    """Daily task to check and send reminders for follow-ups and upcoming interviews"""
    try:
        # Ensure the database is loaded
        agent.load_database()
        
        # Get the current date
        today = datetime.now().date()
        
        # For each user in the database
        for user_id, user_data in agent.db.items():
            try:
                # Try to get the user object from Discord
                user = await bot.fetch_user(int(user_id))
                
                # Check for upcoming interviews in the next 2 days
                upcoming_reminder = ""
                
                for company, roles in user_data["jobs"].items():
                    for role, statuses in roles.items():
                        # Skip if rejected or offered
                        if "rejected" in statuses or "offer" in statuses:
                            continue
                        
                        # Find interview stages
                        for stage, date_str in statuses.items():
                            if stage in ["oa", "phone", "superday"]:
                                # Ensure date_str is a string
                                if not isinstance(date_str, str):
                                    logger.warning(f"Invalid date format for {user_id}, {company}, {role}, {stage}: {date_str}")
                                    continue
                                    
                                interview_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                days_until = (interview_date - today).days
                                
                                # If interview is tomorrow or in 2 days
                                if 0 <= days_until <= 2:
                                    reminder = f"📅 **Reminder:** You have a {stage} interview for the {role} role at {company} on {date_str}"
                                    if days_until == 0:
                                        reminder += " (Today)"
                                    elif days_until == 1:
                                        reminder += " (Tomorrow)"
                                    else:
                                        reminder += f" (In {days_until} days)"
                                        
                                    upcoming_reminder += reminder + "\n"
                
                # Check for needed follow-ups
                followup_reminder = ""
                followup_thresholds = {
                    "applied": 14,  # 2 weeks after application
                    "oa": 7,        # 1 week after online assessment
                    "phone": 7,     # 1 week after phone interview
                    "superday": 7   # 1 week after superday
                }
                
                for company, roles in user_data["jobs"].items():
                    for role, statuses in roles.items():
                        # Skip if rejected or offered
                        if "rejected" in statuses or "offer" in statuses:
                            continue
                            
                        # Get the latest status
                        latest_status = None
                        latest_date = None
                        
                        for status, date_str in statuses.items():
                            # Ensure date_str is a string
                            if not isinstance(date_str, str):
                                logger.warning(f"Invalid date format for {user_id}, {company}, {role}, {status}: {date_str}")
                                continue
                                
                            status_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if latest_date is None or status_date > latest_date:
                                latest_status = status
                                latest_date = status_date
                        
                        # If the latest action was a while ago, suggest follow-up
                        if latest_status in followup_thresholds:
                            days_since = (today - latest_date).days
                            threshold = followup_thresholds[latest_status]
                            
                            # Only remind for new follow-ups (exactly at threshold day)
                            if days_since == threshold:
                                followup_reminder += f"📧 **Follow-up Needed:** It's been {days_since} days since your {latest_status} for the {role} role at {company}. Consider sending a follow-up email.\n"
                
                # Send combined reminders if any exist
                combined_reminder = ""
                if upcoming_reminder:
                    combined_reminder += upcoming_reminder + "\n"
                if followup_reminder:
                    combined_reminder += followup_reminder
                    
                if combined_reminder:
                    try:
                        await user.send(combined_reminder)
                        logger.info(f"Sent reminder to {user.name}")
                    except discord.Forbidden:
                        logger.warning(f"Could not send DM to {user.name}. User may have DMs disabled.")
                    except Exception as e:
                        logger.error(f"Error sending reminder to {user.name}: {e}")
                        
            except discord.NotFound:
                logger.warning(f"User with ID {user_id} not found")
            except Exception as e:
                logger.error(f"Error processing reminders for user {user_id}: {e}")
                
    except Exception as e:
        logger.error(f"Error in reminder task: {e}")


@check_for_reminders.before_loop
async def before_reminder_task():
    """Wait until the bot is ready before starting the reminder task"""
    await bot.wait_until_ready()


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot:
        return

    # Process the message with the agent
    #logger.info(f"Processing message from {message.author}: {message.content}")
    
    try:
        # Show typing indicator while processing
        async with message.channel.typing():
            response = await agent.run(message)
        
        # Split response if too long
        if len(response) > 2000:
            parts = [response[i:i+2000] for i in range(0, len(response), 2000)]
            for part in parts:
                await message.reply(part)
        else:
            await message.reply(response)
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await message.reply("Sorry, I encountered an error while processing your request. Please try again later.")


@bot.command(name="help", help="Shows the job tracker help message")
async def help_command(ctx):
    """Send the job tracker help information."""
    await ctx.send(BOT_HELP_MESSAGE)


# Add passthrough commands for Discord to recognize commands handled by the Mistral agent
@bot.command(name="process")
async def process_command(ctx, *args):
    """Pass the process command to the agent"""
    # Let the agent handle this command
    # The on_message event will have already passed this to the agent
    # We're just registering the command so Discord recognizes it
    pass

@bot.command(name="delete")
async def delete_command(ctx, *args):
    """Pass the delete command to the agent"""
    # Let the agent handle this command
    pass

@bot.command(name="list")
async def list_command(ctx):
    """Pass the list command to the agent"""
    # Let the agent handle this command
    pass

@bot.command(name="upcoming")
async def upcoming_command(ctx):
    """Pass the upcoming command to the agent"""
    # Let the agent handle this command
    pass

@bot.command(name="followups")
async def followups_command(ctx):
    """Pass the followups command to the agent"""
    # Let the agent handle this command
    pass


# Add a command for job insights
@bot.command(name="insights", help="Analyze job automation and get detailed insights")
async def insights_command(ctx, *, content=""):
    """Analyze a job posting for automation risk and provide detailed insights."""
    if not content:
        await ctx.send("Please provide a job description or URL to analyze. Example: `!insights https://example.com/job` or `!insights [paste job description here]`")
        return
        
    # Check if the content is a URL or text
    is_url = content.strip().startswith("http://") or content.strip().startswith("https://")
    
    # Send initial response
    await ctx.send("Analyzing job information... this may take a moment.")
    
    try:
        # Run the job analyzer
        results = analyze_job_automation(content, ONET_TASK_MAPPINGS_FILE)
        
        # Format results using the formatter from new.py
        formatted_output = format_results_for_discord(results)
        
        # Send the formatted output
        if len(formatted_output) > 2000:
            chunks = [formatted_output[i:i+1900] for i in range(0, len(formatted_output), 1900)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(formatted_output)
            
    except Exception as e:
        await ctx.send(f"Error analyzing job: {str(e)}")
        logger.error(f"Job insights error: {e}")





# Add command line functionality
def main():
    """Entry point for the program"""
    # Check if terminal mode is explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--terminal":
        simulate_discord()
    else:
        # Default to Discord mode
        bot.run(token)


if __name__ == "__main__":
    main()