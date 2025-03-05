import os
import discord
import logging
import asyncio
from datetime import datetime, timedelta

from discord.ext import commands, tasks
from dotenv import load_dotenv
from agent import MistralAgent
from insights.job_analyzer import analyze_job_automation
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

❓ **Help**
`!help` - Show this help message

*The bot will also respond to natural language questions about your job search!*
"""

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
                
                for company, statuses in user_data["jobs"].items():
                    # Skip if rejected or offered
                    if "rejected" in statuses or "offer" in statuses:
                        continue
                    
                    # Find interview stages
                    for stage, date_str in statuses.items():
                        if stage in ["oa", "phone", "superday"]:
                            interview_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            days_until = (interview_date - today).days
                            
                            # If interview is tomorrow or in 2 days
                            if 0 <= days_until <= 2:
                                reminder = f"📅 **Reminder:** You have a {stage} interview with {company} on {date_str}"
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
                
                for company, statuses in user_data["jobs"].items():
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
                    
                    # If the latest action was a while ago, suggest follow-up
                    if latest_status in followup_thresholds:
                        days_since = (today - latest_date).days
                        threshold = followup_thresholds[latest_status]
                        
                        # Only remind for new follow-ups (exactly at threshold day)
                        if days_since == threshold:
                            followup_reminder += f"📧 **Follow-up Needed:** It's been {days_since} days since your {latest_status} with {company}. Consider sending a follow-up email.\n"
                
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
    logger.info(f"Processing message from {message.author}: {message.content}")
    
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
    # Update the help message to include job analysis
    help_message = BOT_HELP_MESSAGE + """

🤖 **Job Analysis**
`!analyze [job description or URL]` - Analyze a job posting for automation risk
`!job [job description or URL]` - Same as analyze command
  • Example: `!analyze https://example.com/job-posting`
  • Example: `!job [paste job description here]`
"""
    await ctx.send(help_message)


# Add a new function to handle job analyzer commands
@bot.command(name="analyze", help="Analyze job automation and career growth potential")
async def analyze_command(ctx, *, content=""):
    """Analyze a job posting for automation risk and career insights."""
    if not content:
        await ctx.send("Please provide a job description or URL to analyze. Example: `!analyze https://example.com/job` or `!analyze [paste job description here]`")
        return
        
    await handle_job_analysis(ctx, content)

# Also add an alternate command name for the same function
@bot.command(name="job", help="Analyze job automation and career growth potential")
async def job_command(ctx, *, content=""):
    """Alias for the analyze command."""
    if not content:
        await ctx.send("Please provide a job description or URL to analyze. Example: `!job https://example.com/job` or `!job [paste job description here]`")
        return
        
    await handle_job_analysis(ctx, content)

# Update the handle_job_analysis to work with ctx instead of message
async def handle_job_analysis(ctx, content):
    """
    Process job analysis requests
    
    Args:
        ctx: The Discord context object
        content: The content of the message
    """
    # Check if the content is a URL or text
    is_url = content.strip().startswith("http://") or content.strip().startswith("https://")
    
    # Send initial response
    await ctx.send("Analyzing job information... this may take a moment.")
    
    try:
        # Run the job analyzer
        results = analyze_job_automation(content, ONET_TASK_MAPPINGS_FILE)
        
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
        
        # Format the response
        response = f"**AUTOMATION RISK ASSESSMENT: {risk_level} ({automation_risk:.2f})**\n"
        response += f"{risk_desc}\n\n"
        
        # Add career growth info
        career = results.get('Career_Growth_Potential', {})
        response += f"**Career Growth Potential: {career.get('level', 'Unknown')}**\n"
        response += f"• {career.get('description', '')}\n"
        response += f"• Outlook: {career.get('outlook', '')}\n"
        response += f"• Skills: {career.get('skill_demand', '')}\n"
        response += f"• Recommendation: {career.get('recommendations', '')}\n\n"
        
        # Add industry stability
        industry = results.get('Industry_Stability', {})
        response += f"**Industry: {results.get('Industry', 'Unknown')}**\n"
        response += f"• Status: {industry.get('status', 'Unknown')}\n"
        response += f"• Trend: {industry.get('trend', '')}\n"
        response += f"• Disruption Risk: {industry.get('disruption_risk', '')}\n"
        
        # Add matched tasks (limited to keep response reasonable)
        if 'Matched_Tasks' in results and results['Matched_Tasks']:
            response += "\n**Job Tasks Analysis:**\n"
            for i, task in enumerate(results['Matched_Tasks'][:3]):  # Limit to 3 tasks
                emoji = "🟢" if task['automation_pct'] < 0.3 else "🟠" if task['automation_pct'] < 0.6 else "🔴"
                response += f"{i+1}. {emoji} '{task['job_task'][:80]}...'\n"
                response += f"   → Automation: {task['automation_pct']*100:.1f}%\n"
        
        # Send the response (potentially breaking into chunks if too long)
        if len(response) > 2000:
            # Split into multiple messages if too long for Discord
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(response)
            
    except Exception as e:
        await ctx.send(f"Error analyzing job: {str(e)}")
        logger.error(f"Job analysis error: {e}")

# Start the bot, connecting it to the gateway
bot.run(token)
