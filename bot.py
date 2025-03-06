import os
import sys
import logging
import discord
from discord.ext import commands
from dotenv import load_dotenv
from insights.job_analyzer import analyze_job_automation, format_results_for_discord
from insights.config import ONET_TASK_MAPPINGS_FILE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

# Load environment variables
load_dotenv()

# Bot configuration
PREFIX = "!"
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

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

def run_discord_bot():
    """Run the Discord bot"""
    # Create bot instance
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=PREFIX, intents=intents)

    @bot.event
    async def on_ready():
        logger.info(f"{bot.user} has connected to Discord!")

    @bot.command(name="analyze", help="Analyze a job posting URL for automation potential")
    async def analyze(ctx, url: str):
        if not url.startswith('http'):
            await ctx.send("Please provide a valid URL starting with http:// or https://")
            return

        try:
            async with ctx.typing():
                # Run the analysis
                results = analyze_job_automation(url, ONET_TASK_MAPPINGS_FILE)
                
                # Format and send results
                if isinstance(results, dict) and 'error' not in results:
                    formatted_output = format_results_for_discord(results)
                    await ctx.send(formatted_output)
                else:
                    await ctx.send(f"Error in analysis: {results.get('error', 'Unknown error occurred')}")
                    
        except Exception as e:
            logger.error(f"Error analyzing job: {e}")
            await ctx.send(f"Error: {str(e)}")

    # Start the bot
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    # Check if terminal mode is explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--terminal":
        simulate_discord()
    else:
        # Default to Discord mode
        run_discord_bot()
