# PingPal: Job Application Tracker & Career Assistant

PingPal is a Discord bot designed to help job seekers track their applications, manage interview schedules, and gain insights about job automation risks. Built with Python, it leverages Mistral AI for natural language processing and provides timely reminders and job market insights.

## Features

### 📋 Job Application Tracking
- **Add & Update Applications**: Track companies, roles, application status, and important dates
- **Multiple Status Options**: Record different stages (applied, online assessment, phone interview, superday, offer, rejected)
- **Application Management**: View, update, and delete job applications easily

### 📊 Application Insights
- **List Applications**: See all your job applications in one organized view
- **Upcoming Interviews**: Get a list of your upcoming interviews with countdown timers
- **Follow-up Reminders**: Receive suggestions on when to follow up based on application stage

### 🤖 Job Market Analysis
- **Automation Risk Analysis**: Analyze job postings to understand automation risk factors
- **Detailed Job Insights**: Get comprehensive reports on job requirements, skills, and market trends

### 🔔 Automated Reminders
- **Interview Alerts**: Receive reminders about upcoming interviews
- **Follow-up Notifications**: Get timely reminders to send follow-up emails

## Commands

### Job Tracking
- `!process company_name role status [date]` - Add or update a job application
  - Status options: applied, oa (online assessment), phone, superday, offer, rejected
  - Date format: YYYY-MM-DD (optional, defaults to today)
  - Example: `!process Google SWE-intern applied 2025-02-15`
- `!delete company_name role` - Delete a job application
  - Example: `!delete Google SWE-intern`

### Viewing Applications
- `!list` - List all your job applications
- `!upcoming` - Show upcoming interviews
- `!followups` - Show applications that need follow-up emails

### Job Analysis
- `!insights [job description or URL]` - Analyze a job posting for automation risk
  - Example: `!insights https://example.com/job-posting`
  - Example: `!insights [paste job description here]`

### Help
- `!help` - Show the help message with all available commands

## Natural Language Interaction
PingPal can respond to natural language questions about your job search and provide advice through the Mistral AI integration.

## Technical Architecture

### Components
- **Discord Bot**: Built with discord.py for the user interface
- **Mistral AI**: Powers natural language understanding and generation
- **Supabase**: Stores user data and job application information
- **Job Analyzer**: Evaluates job postings for automation risk and provides insights

### Data Storage
- User data is stored in Supabase with the following structure:
  - User ID (Discord user identifier)
  - Job applications with companies, roles, and status history
  - Application dates and follow-up schedules

Happy job hunting with PingPal! 🚀
