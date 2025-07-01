# 🔗 Slack Integration Guide

**Complete guide for integrating the Hardware Digest System with your Slack workspace**

## Overview

This guide will walk you through connecting your Hardware Digest System to your actual Slack workspace, enabling real-time monitoring and analysis of your hardware team's communications.

## Prerequisites

- ✅ Hardware Digest System running (see main README)
- ✅ Slack workspace administrator access
- ✅ Your team's hardware channels identified

## 🚀 Quick Setup

### Step 1: Create Your Slack App

1. **Go to Slack API Console**
   - Visit [https://api.slack.com/apps](https://api.slack.com/apps)
   - Click **"Create New App"**

2. **Choose App Creation Method**
   - Select **"From scratch"**
   - App Name: `Hardware Digest Bot`
   - Choose your workspace: `[Your Hardware Team Workspace]`
   - Click **"Create App"**

### Step 2: Configure Bot Permissions

Navigate to **OAuth & Permissions** in your app settings and add these **Bot Token Scopes**:

#### Required Scopes
```
channels:read       # Read channel information and membership
channels:history    # Read message history from public channels
chat:write         # Post digest summaries and alerts
chat:write.public  # Post in channels bot isn't member of
```

#### Recommended Scopes
```
users:read         # Read user info for @mention analysis
files:read         # Access hardware specs and design files
reactions:read     # Monitor reactions for sentiment analysis
groups:read        # Access private channel information
groups:history     # Read private channel message history
```

#### Optional Advanced Scopes
```
channels:manage    # Create channels for specific alerts
pins:read         # Access pinned important messages
bookmarks:read    # Access bookmarked resources
```

### Step 3: Enable Event Subscriptions

1. **Go to Event Subscriptions**
2. **Enable Events**: Toggle to "On"
3. **Request URL**: `https://your-domain.com/api/slack/events`
   - For development: Use ngrok or similar tunnel service
   - For production: Your actual domain

4. **Subscribe to Bot Events**:
```
message.channels    # New messages in public channels
message.groups      # New messages in private channels
reaction_added      # Team reactions to messages
reaction_removed    # Reaction changes for sentiment tracking
file_shared        # Hardware specs/designs shared
app_mention        # When bot is mentioned directly
```

### Step 4: Install App & Get Tokens

1. **Install App**
   - Go to **"Install App"** section
   - Click **"Install to Workspace"**
   - Review permissions and click **"Allow"**

2. **Copy Bot Token**
   - Copy the **Bot User OAuth Token** (starts with `xoxb-`)
   - This is your `SLACK_BOT_TOKEN`

3. **Get App-Level Token** (for Socket Mode)
   - Go to **"Basic Information"**
   - Scroll to **"App-Level Tokens"**
   - Click **"Generate Token and Scopes"**
   - Name: `socket_mode_token`
   - Scope: `connections:write`
   - Copy the token (starts with `xapp-`)

4. **Get Signing Secret**
   - In **"Basic Information"**
   - Copy the **Signing Secret**
   - This is your `SLACK_SIGNING_SECRET`

### Step 5: Configure Socket Mode (Recommended for Development)

1. **Enable Socket Mode**
   - Go to **"Socket Mode"**
   - Toggle **"Enable Socket Mode"** to On
   - Select your app-level token

2. **Benefits of Socket Mode**
   - No need for public webhook URLs
   - Perfect for development/testing
   - Real-time bidirectional communication

### Step 6: Update Environment Configuration

Update your `.env` file with the actual tokens:

```bash
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-actual-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-level-token-here
SLACK_SIGNING_SECRET=your-actual-signing-secret-here

# Slack Integration Settings
SLACK_USE_SOCKET_MODE=true
SLACK_CHANNELS=hardware-team,supply-chain,quality-assurance,product-launch
SLACK_ALERT_CHANNEL=hardware-alerts
SLACK_DIGEST_CHANNEL=daily-digest

# Slack Monitoring Configuration
SLACK_MONITOR_REACTIONS=true
SLACK_MONITOR_FILES=true
SLACK_MONITOR_THREADS=true
SLACK_MONITOR_MENTIONS=true
```

### Step 7: Add Bot to Channels

Invite your bot to the relevant channels:

```bash
# In each target channel, type:
/invite @Hardware Digest Bot

# Or use the channel management interface
# to add the bot to multiple channels at once
```

**Recommended Channels:**
- `#hardware-team` - Main hardware discussions
- `#supply-chain` - Supplier and component discussions
- `#quality-assurance` - Testing and quality issues
- `#product-launch` - GTM and launch planning
- `#manufacturing` - Production and manufacturing updates
- `#hardware-alerts` - Dedicated channel for bot alerts

## 🔧 Advanced Configuration

### Production Webhook Setup

For production deployments, use webhooks instead of Socket Mode:

```bash
# Production .env settings
SLACK_USE_SOCKET_MODE=false
SLACK_WEBHOOK_URL=https://yourdomain.com/api/slack/events
SLACK_WEBHOOK_VERIFICATION=true
```

### Custom Message Filtering

Configure which message types to analyze:

```bash
# Message Analysis Configuration
SLACK_ANALYZE_THREADS=true
SLACK_ANALYZE_EDITS=true
SLACK_ANALYZE_REACTIONS=true
SLACK_ANALYZE_FILES=true
SLACK_IGNORE_BOTS=true
SLACK_IGNORE_CHANNELS=random,general,off-topic

# Keyword Filtering
SLACK_SUPPLY_CHAIN_KEYWORDS=delay,shortage,supplier,vendor,component,chip,part
SLACK_QUALITY_KEYWORDS=defect,yield,test,failure,issue,bug,problem,rework
SLACK_TIMELINE_KEYWORDS=deadline,milestone,schedule,timeline,delay,blocked,urgent
```

### Risk Alert Configuration

Set up automated alerts for critical issues:

```bash
# Alert Configuration
SLACK_ENABLE_ALERTS=true
SLACK_ALERT_THRESHOLD=high
SLACK_ALERT_COOLDOWN=3600  # 1 hour cooldown between similar alerts
SLACK_ALERT_MENTIONS=@channel,@here  # Mention level for critical alerts

# Alert Templates
SLACK_SUPPLY_CHAIN_ALERT_TEMPLATE="🚨 Supply Chain Alert: {issue} - Impact: {impact}"
SLACK_QUALITY_ALERT_TEMPLATE="⚠️ Quality Issue Detected: {issue} - Severity: {severity}"
SLACK_TIMELINE_ALERT_TEMPLATE="📅 Timeline Risk: {issue} - Deadline Impact: {impact}"
```

## 🧪 Testing Your Integration

### 1. Test Bot Presence

```bash
# In a monitored channel, type:
@Hardware Digest Bot status

# Expected response:
# ✅ Hardware Digest Bot is online and monitoring this channel
# 📊 Analyzed 1,247 messages today
# ⚠️ 3 active risk alerts
# 🔗 Dashboard: https://your-dashboard.com
```

### 2. Test Message Analysis

Send test messages to verify analysis:

```bash
# Supply Chain Test
"Our chip supplier just reported a 2-week delay on the main processor order"

# Quality Test  
"Manufacturing yield dropped to 75% this morning, investigating root cause"

# Timeline Test
"We're behind schedule on firmware testing, may need to push launch date"
```

### 3. Verify Dashboard Integration

1. Check the dashboard at `http://localhost:8501`
2. Verify new messages appear in "Latest Messages"
3. Confirm risk assessments are generated
4. Test daily digest generation

## 📊 Expected Behavior

### Real-Time Monitoring
- 📡 Bot monitors all configured channels 24/7
- 🤖 Messages analyzed within seconds of posting
- 📈 Risk scores updated in real-time
- 🎯 Relevant messages categorized automatically

### Daily Digest Posts
Every morning at 9 AM (configurable), the bot posts:

```
🚀 Hardware Team Digest - July 1, 2025

⚠️ CRITICAL RISKS DETECTED:
• Supply Chain: Component X delivery delayed 2 weeks
• Quality: 15% yield drop in manufacturing line B  
• Timeline: Product launch may slip by 1 week

📊 TEAM ACTIVITY:
• 47 messages analyzed across 4 channels
• 12 action items identified
• 3 blocked dependencies found

🎯 TOP PRIORITIES:
• Review supplier backup options for Component X
• Investigate yield drop in manufacturing line B
• Assess launch timeline impact and mitigation plans

🔗 Full Analysis: http://your-dashboard.com/digest/2025-07-01
```

### Alert Notifications
Critical issues trigger immediate alerts:

```
🚨 URGENT: Supply Chain Alert 🚨

Issue: Primary chip supplier reporting 3-week delay
Impact: High - affects Q3 launch timeline
Confidence: 95%

Recommended Actions:
• Contact backup suppliers immediately
• Assess redesign options for alternative chips
• Update stakeholders on potential timeline impact

🔗 Details: http://your-dashboard.com/alert/sc-2025-07-01-001
```

## 🔍 Troubleshooting

### Common Issues

#### Bot Not Responding
1. **Check Token Validity**
   ```bash
   curl -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
        https://slack.com/api/auth.test
   ```

2. **Verify Permissions**
   - Ensure bot has required scopes
   - Check if bot is in the channel

#### Messages Not Being Analyzed
1. **Check Channel Configuration**
   ```bash
   # Verify SLACK_CHANNELS in .env
   echo $SLACK_CHANNELS
   ```

2. **Check Bot Membership**
   - Bot must be invited to channels
   - Verify bot appears in channel member list

#### Webhooks Not Working
1. **Verify Webhook URL**
   - Test endpoint: `curl -X POST https://your-domain.com/api/slack/events`
   - Check SSL certificate validity

2. **Check Signing Secret**
   - Verify SLACK_SIGNING_SECRET matches app settings

### Debug Mode

Enable debug logging:

```bash
# Add to .env
SLACK_DEBUG=true
SLACK_LOG_LEVEL=debug
LOG_SLACK_MESSAGES=true
```

## 📚 API Endpoints

### Slack Integration Endpoints

```bash
# Check Slack connection status
GET /api/slack/status

# Manual message analysis
POST /api/slack/analyze
{
  "channel": "hardware-team",
  "message": "Sample message to analyze"
}

# Get channel statistics
GET /api/slack/channels/{channel_id}/stats

# Force digest generation
POST /api/slack/digest/generate

# Test alert system
POST /api/slack/alerts/test
```

## 🚀 Going Live

### Pre-Launch Checklist

- [ ] Slack app created and configured
- [ ] Bot tokens added to `.env`
- [ ] Bot invited to all target channels
- [ ] Test messages analyzed successfully
- [ ] Daily digest scheduled and tested
- [ ] Alert system configured and tested
- [ ] Team notified about new bot

### Launch Day

1. **Announce to Team**
   ```
   🎉 Introducing Hardware Digest Bot!
   
   Our new AI assistant will help us stay on top of:
   • Supply chain risks and delays
   • Quality issues and trends
   • Timeline management and blockers
   
   The bot is now monitoring our channels and will provide:
   • Daily digest summaries every morning
   • Real-time alerts for critical issues
   • Trend analysis and insights
   
   Dashboard: http://your-dashboard.com
   Questions? Ask in #hardware-team
   ```

2. **Monitor Initial Performance**
   - Watch for any errors in logs
   - Verify message analysis accuracy
   - Collect team feedback
   - Adjust sensitivity settings as needed

### Post-Launch Optimization

- **Week 1**: Fine-tune keyword filters and alert thresholds
- **Week 2**: Adjust digest content based on team feedback
- **Month 1**: Analyze effectiveness and add new features
- **Ongoing**: Regular review of patterns and improvements

## 🆘 Support

### Getting Help

1. **Check Logs**
   ```bash
   docker-compose logs fastapi | grep -i slack
   ```

2. **API Health Check**
   ```bash
   curl http://localhost:8000/api/slack/status
   ```

3. **Community Support**
   - GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
   - Documentation: [Wiki](https://github.com/your-repo/wiki)
   - Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**🎯 Ready to transform your hardware team's communication intelligence!**

*This integration typically takes 15-30 minutes to complete and provides immediate value to your GTM process.* 