# CS 153 - Infrastructure at Scale AI Agent Starter Code

### Join the Discord Server

First, every member of the team should join the Discord server using the invite link on Ed.

### Creating/Joining Your Group Channel

How to create or join your group channel:

1. Send a Direct Message (DM) to the Admin Bot.
2. Pick a **unique** group name (**IMPORTANT**)
3. Use the following command format:`.channel <Group Name>`
4. Replace `<Group Name>` with the name of your project group (e.g., `.channel Group 1`).

**What Happens When You Use the Command:**

If the Channel Already Exists:

- Check if you already have the role for this group. If you don’t have the role, it will assign you the role corresponding to `<Group Name>` granting you access to the channel.

If the Channel Does Not Exist:

- Create a new text channel named `<Group-Name>` in the Project Channels category.
- Create a role named `<group name>` (the system will intentionally lower the case) and assign it to you.

- Set permissions so that:
  - Only members with the `<group name>` role can access the channel.
  - The app and server admins have full access. All other server members are denied access.
  - Once completed, you'll be able to access your group's private channel in the Project Channels category.

## [One student per group] Setting up your bot

##### Note: only ONE student per group should follow the rest of these steps.

### Download files

1. Fork and clone this GitHub repository.
2. Share the repo with your teammates.
3. Create a file called `.env` the same directory/folder as `bot.py`. The `.env` file should look like this, replacing the “your key here” with your key. In the below sections, we explain how to obtain Discord keys and Mistral API keys.

```
DISCORD_TOKEN=“your key here”
MISTRAL_API_KEY=“your key here”
```

#### Making the bot

1. Go to https://discord.com/developers and click “New Application” in the top right corner.
2. Pick a cool name for your new bot!

##### It is very important that you name your app exactly following this scheme; some parts of the bot’s code rely on this format.

1. Next, you’ll want to click on the tab labeled “Bot” under “Settings.”
2. Click “Copy” to copy the bot’s token. If you don’t see “Copy”, hit “Reset Token” and copy the token that appears (make sure you’re the first team member to go through these steps!)
3. Open `.env` and paste the token between the quotes on the line labeled `DISCORD_TOKEN`.
4. Scroll down to a region called “Privileged Gateway Intents”
5. Tick the options for “Presence Intent”, “Server Members Intent”, and “Message Content Intent”, and save your changes.
6. Click on the tab labeled “OAuth2” under “Settings”
7. Locate the tab labeled “OAuth2 URL Generator” under “OAuth2”. Check the box labeled “bot”. Once you do that, another area with a bunch of options should appear lower down on the page.
8. Check the following permissions, then copy the link that’s generated. <em>Note that these permissions are just a starting point for your bot. We think they’ll cover most cases, but you may run into cases where you want to be able to do more. If you do, you’re welcome to send updated links to the teaching team to re-invite your bot with new permissions.</em>
  <img width="1097" alt="bot_permissions" src="https://github.com/user-attachments/assets/4db80209-e8d3-4e71-8cff-5f5e04beceeb" />
9. Copy paste this link into the #app-invite-link channel on the CS 153 Discord server. Someone in the teaching team will invite your bot.
10. After your bot appears in #welcome, find your bot's "application ID" on the Discord Developer panel.

![CleanShot 2025-01-21 at 23 42 53@2x](https://github.com/user-attachments/assets/2cf6b8fd-5756-494c-a6c3-8c61e821d568)
    
12. Send a DM to the admin bot: use the `.add-bot <application ID>` command to add the bot to your channel.

#### Setting up the Mistral API key

1. Go to [Mistral AI Console](https://console.mistral.ai) and sign up for an account. During sign-up, you will be prompted to set up a workspace. Choose a name for your workspace and select "I'm a solo creator." If you already have an account, log in directly.
2. After logging in, navigate to the "Workspace" section on the left-hand menu. Click on "Billing" and select “Experiment for free”.
3. A pop-up window will appear. Click "Accept" to subscribe to the experiment plan and follow the instructions to verify your phone number. After verifying your phone number, you may need to click "Experiment for free" again to finish subscribing. 
4. Once you have successfully subscribed to the experiment plan, go to the "API keys" page under the “API” section in the menu on the left.
5. Click on "Create new key" to generate a new API key.
6. After the key is generated, it will appear under “Your API keys” with the text: `“Your key is: <your-api-key>”`. Copy the API key and save it securely, as it will not be displayed again for security reasons.
7. Open your `.env` file and paste the API key between the quotes on the line labeled `MISTRAL_API_KEY`.

#### Setting up the starter code

We'll be using Python, if you've got a good Python setup already, great! But make sure that it is at least Python version 3.8. If not, the easiest thing to do is to make sure you have at least 3GB free on your computer and then to head over to [miniconda install](https://docs.anaconda.com/miniconda/install/) and install the Python 3 version of Anaconda. It will work on any operating system.

After you have installed conda, close any open terminals you might have. Then open a terminal in the same folder as your `bot.py` file (If you haven’t used your terminal before, check out [this guide](https://www.macworld.com/article/2042378/master-the-command-line-navigating-files-and-folders.html)!). Once in, run the following command

## 1. Create an environment with dependencies specified in env.yml:
    conda env create -f environment.yml

## 2. Activate the new environment:
    conda activate discord_bot
    
This will install the required dependencies to start the project.