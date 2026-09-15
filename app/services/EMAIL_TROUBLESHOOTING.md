# Email Service Troubleshooting Guide

## Current Issue: Connection Timeout

**Problem:** The SMTP server `mail.andalusiagroup.net` is not reachable from your current network location.

**Diagnosis Results:**
- ❌ Hostname resolves to IP: 3.72.115.52
- ❌ Server not responding to ping
- ❌ All SMTP ports (25, 587, 465) timing out
- ✅ Network connectivity appears functional

## Solutions

### 1. Connect to VPN (RECOMMENDED)

Internal mail servers are typically only accessible from within the corporate network.

```bash
# Connect to your company VPN first, then run the application
# This is the most common solution for "connection timeout" issues
```

### 2. Test Connection After VPN

Once connected to VPN, verify connectivity:

```bash
# Test if server is reachable
ping mail.andalusiagroup.net

# Test if SMTP port is open
nc -zv mail.andalusiagroup.net 587

# Run the connectivity test
python3 mail_client.py
```

### 3. Use Alternative SMTP for Testing

If you need to test the email functionality without VPN access:

**Option A: Gmail**
1. Edit `mail_config.py`
2. Change `ACTIVE_CONFIG = GMAIL_CONFIG`
3. Set up Gmail App Password: https://myaccount.google.com/apppasswords
4. Update GMAIL_CONFIG with your credentials

**Option B: Outlook/Office365**
1. Edit `mail_config.py`
2. Change `ACTIVE_CONFIG = OUTLOOK_CONFIG`
3. Update OUTLOOK_CONFIG with your credentials

### 4. SSH Tunnel (Advanced)

If you have SSH access to an internal server:

```bash
# Create tunnel
ssh -L 2525:mail.andalusiagroup.net:25 user@internal-gateway

# In mail_config.py, create new config:
TUNNEL_CONFIG = {
    "SMTP_SERVER": "localhost",
    "SMTP_PORT": 2525,
    "SMTP_USER": "andalusia\\rafik.atallah",
    "SMTP_PASSWORD": "refa2001",
    "SENDER_EMAIL": "rafik.atallah@andalusiagroup.net",
    "USE_TLS": False,
    "USE_SSL": False,
}
```

### 5. Request IT Support

Contact IT to:
- Verify SMTP server is running
- Check if firewall rules allow outbound SMTP from your IP
- Confirm correct SMTP server hostname
- Enable SMTP AUTH on Exchange server if needed

## Quick Configuration Check

Run the test script to verify your setup:

```bash
python3 mail_client.py
```

This will:
1. Test connectivity on ports 25, 587, and 465
2. Show which configuration works
3. Prompt to send a test email

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `TimeoutError: timed out` | Server not reachable | Connect to VPN |
| `ConnectionRefusedError` | Port blocked/closed | Try different port (25/587/465) |
| `SMTPAuthenticationError` | Wrong credentials | Check username/password format |
| `[Errno -2] Name or service not known` | DNS failure | Check hostname spelling |
| `SMTPServerDisconnected` | Connection dropped | Check firewall/network stability |

## Current Configuration

Edit `mail_config.py` to change settings:

```python
# Switch between configs
ACTIVE_CONFIG = INTERNAL_CONFIG  # For production (requires VPN)
# ACTIVE_CONFIG = GMAIL_CONFIG   # For testing with Gmail
# ACTIVE_CONFIG = OUTLOOK_CONFIG # For testing with Outlook
```

## Security Notes

⚠️ **IMPORTANT**: The current configuration has credentials in plain text.

**For Production:**
1. Move credentials to environment variables
2. Use `.env` file (add to `.gitignore`)
3. Consider using OAuth2 for Gmail/Outlook
4. Use secrets management service (AWS Secrets Manager, Azure Key Vault)

Example with environment variables:

```python
import os

SMTP_USER = os.getenv("SMTP_USER", "default_user")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "default_pass")
```

## Testing Checklist

- [ ] Can ping mail.andalusiagroup.net
- [ ] SMTP port is accessible (nc -zv test)
- [ ] VPN is connected (if required)
- [ ] Credentials are correct
- [ ] SMTP AUTH is enabled on server
- [ ] Firewall allows outbound SMTP
- [ ] Using correct authentication format (domain\\user vs user@domain)
