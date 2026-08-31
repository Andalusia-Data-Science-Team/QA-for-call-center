"""
SMTP Configuration for Email Service

ISSUE: mail.andalusiagroup.net is not reachable from this network.
SOLUTIONS:

1. VPN Connection Required
   - Connect to company VPN before running the application
   - Internal mail servers are often only accessible via VPN

2. Use Alternative SMTP Service (for testing/development)
   - Gmail, Outlook, or other external SMTP
   - See configurations below

3. SSH Tunnel (if you have access to internal server)
   - ssh -L 2525:mail.andalusiagroup.net:25 user@internal-server
   - Then use localhost:2525 as SMTP server

4. Request Firewall Exception
   - Contact IT to allow outbound SMTP from this server
"""

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION (Internal Exchange) - Requires VPN
# ═══════════════════════════════════════════════════════════════════
INTERNAL_CONFIG = {
    "SMTP_SERVER": "mail.andalusiagroup.net",
    "SMTP_PORT": 587,
    "SMTP_USER": "andalusia\\rafik.atallah",
    "SMTP_PASSWORD": "refa2001",
    "SENDER_EMAIL": "rafik.atallah@andalusiagroup.net",
    "USE_TLS": True,
    "USE_SSL": False,
}

# ═══════════════════════════════════════════════════════════════════
# ALTERNATIVE: Gmail SMTP (for testing)
# ═══════════════════════════════════════════════════════════════════
# NOTE: Requires App Password (not your regular Gmail password)
# Create one at: https://myaccount.google.com/apppasswords
GMAIL_CONFIG = {
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
    "SMTP_USER": "your.email@gmail.com",
    "SMTP_PASSWORD": "your-app-password-here",
    "SENDER_EMAIL": "your.email@gmail.com",
    "USE_TLS": True,
    "USE_SSL": False,
}

# ═══════════════════════════════════════════════════════════════════
# ALTERNATIVE: Outlook/Office365 SMTP
# ═══════════════════════════════════════════════════════════════════
OUTLOOK_CONFIG = {
    "SMTP_SERVER": "smtp-mail.outlook.com",
    "SMTP_PORT": 587,
    "SMTP_USER": "your.email@outlook.com",
    "SMTP_PASSWORD": "your-password-here",
    "SENDER_EMAIL": "your.email@outlook.com",
    "USE_TLS": True,
    "USE_SSL": False,
}

# ═══════════════════════════════════════════════════════════════════
# ACTIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
# Change this to switch between configs
# Options: INTERNAL_CONFIG, GMAIL_CONFIG, OUTLOOK_CONFIG

ACTIVE_CONFIG = INTERNAL_CONFIG  # <-- Change this when needed

# Export active config
SMTP_SERVER = ACTIVE_CONFIG["SMTP_SERVER"]
SMTP_PORT = ACTIVE_CONFIG["SMTP_PORT"]
SMTP_USER = ACTIVE_CONFIG["SMTP_USER"]
SMTP_PASSWORD = ACTIVE_CONFIG["SMTP_PASSWORD"]
SENDER_EMAIL = ACTIVE_CONFIG["SENDER_EMAIL"]
USE_TLS = ACTIVE_CONFIG["USE_TLS"]
USE_SSL = ACTIVE_CONFIG["USE_SSL"]
