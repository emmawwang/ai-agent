import os
import discord
import logging
import asyncio
from datetime import datetime, timedelta

from discord.ext import commands, tasks
from dotenv import load_dotenv
from agent import MistralAgent

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
`!process company_name status [date]` - Add or update a job application
  • Status options: applied, oa, phone, superday, offer, rejected
  • Date format: YYYY-MM-DD (optional, defaults to today)
  • Example: `!process Google applied 2025-02-15`

📊 **View Your Applications**
`!list` - List all your job applications
`!upcoming` - Show upcoming interviews
`!followups` - Show applications that need follow-up emails

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
    await ctx.send(BOT_HELP_MESSAGE)


# Start the bot, connecting it to the gateway
bot.run(token)
