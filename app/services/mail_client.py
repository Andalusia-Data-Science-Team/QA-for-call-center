import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import configuration from separate config file
try:
    from mail_config import (
        SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, 
        SENDER_EMAIL, USE_TLS, USE_SSL
    )
except ImportError:
    # Fallback to inline config if mail_config.py not found
    SMTP_SERVER = "mail.andalusiagroup.net"
    SMTP_PORT = 587
    SENDER_EMAIL = "rafik.atallah@andalusiagroup.net"
    SMTP_USER = "andalusia\\rafik.atallah"
    SMTP_PASSWORD = "refa2001"
    USE_TLS = True
    USE_SSL = False


def build_html_body(body_text: str, review_link: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Review Request</title>
</head>

<body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">

<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f4f4f4;padding:40px 0;">
<tr>
<td align="center">

<table role="presentation"
       width="650"
       cellspacing="0"
       cellpadding="0"
       border="0"
       style="background:#ffffff;border-radius:10px;border:1px solid #e5e5e5;">

    <!-- Header -->
    <tr>
        <td style="background:#0078D4;padding:25px;border-radius:10px 10px 0 0;">
            <h2 style="margin:0;color:white;font-weight:600;">
                QA for Agent Performance
            </h2>
        </td>
    </tr>

    <!-- Body -->
    <tr>
        <td style="padding:35px;">

            <p style="font-size:16px;color:#333;margin-top:0;">
                Hello,
            </p>

            <p style="font-size:15px;color:#555;line-height:1.7;">
                {body_text}
            </p>

            <p style="font-size:15px;color:#555;line-height:1.7;">
                Once you have completed your review, please click the button below.
            </p>

            <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" style="margin:35px auto;">
                <tr>
                    <td bgcolor="#107C10" style="border-radius:6px;">
                        <a href="{review_link}"
                           target="_blank"
                           style="
                               display:inline-block;
                               padding:14px 36px;
                               font-size:16px;
                               font-weight:bold;
                               font-family:Arial,Helvetica,sans-serif;
                               color:#ffffff;
                               text-decoration:none;
                               background:#107C10;
                               border-radius:6px;">
                            ✔ Reviewed
                        </a>
                    </td>
                </tr>
            </table>

            <p style="font-size:13px;color:#888;">
                If the button does not work, copy and paste the following link into your browser:
            </p>

            <p style="word-break:break-all;">
                <a href="{review_link}" style="color:#0078D4;">
                    {review_link}
                </a>
            </p>

        </td>
    </tr>

    <!-- Footer -->
    <tr>
        <td style="padding:20px 35px;background:#fafafa;border-top:1px solid #e6e6e6;">

            <p style="margin:0;font-size:12px;color:#888;">
                This is an automated message. Please do not reply.
            </p>

            <p style="margin-top:8px;font-size:12px;color:#888;">
                © 2026 Your Company
            </p>

        </td>
    </tr>

</table>

</td>
</tr>
</table>

</body>
</html>
"""


def send_email(to: str, subject: str, body_text: str, review_link: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = to

    html_part = MIMEText(build_html_body(body_text, review_link), "html")
    msg.attach(html_part)

    try:
        print(f"🔄 Attempting to connect to {SMTP_SERVER}:{SMTP_PORT}...")
        
        # Use SMTP_SSL for port 465, regular SMTP for others
        if USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        
        with server:
            server.set_debuglevel(0)  # Set to 1 for verbose debugging
            
            print("📝 Sending EHLO...")
            server.ehlo()
            
            # Use STARTTLS if enabled and not already using SSL
            if USE_TLS and not USE_SSL:
                if server.has_extn('STARTTLS'):
                    print("🔒 Starting TLS...")
                    server.starttls()
                    server.ehlo()  # EHLO again after STARTTLS
                else:
                    print("⚠️  STARTTLS not available, proceeding without encryption")
            
            print(f"🔑 Logging in as {SMTP_USER}...")
            server.login(SMTP_USER, SMTP_PASSWORD)
            
            print(f"📧 Sending email to {to}...")
            server.sendmail(SENDER_EMAIL, to, msg.as_string())
            print(f"✅ Email sent successfully to {to}")

    except smtplib.SMTPConnectError as e:
        print(f"❌ Connection Error: Could not connect to {SMTP_SERVER}:{SMTP_PORT}")
        print(f"   Details: {e}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Authentication Error: Invalid username or password")
        print(f"   Details: {e}")
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {e}")
    except TimeoutError as e:
        print(f"❌ Timeout Error: Connection timed out after 30 seconds")
        print(f"   Check if {SMTP_SERVER}:{SMTP_PORT} is accessible from your network")
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")


# ── Helper function to test SMTP connectivity ──────────────────────
def test_smtp_connection():
    """Test different SMTP configurations to find what works"""
    configs = [
        {"port": 25, "tls": False, "desc": "Port 25 (Plain/Relay)"},
        {"port": 587, "tls": True, "desc": "Port 587 (STARTTLS)"},
        {"port": 465, "ssl": True, "desc": "Port 465 (SSL/TLS)"},
    ]
    
    print("\n" + "="*60)
    print("Testing SMTP connectivity...")
    print("="*60 + "\n")
    
    for config in configs:
        try:
            print(f"🔍 Testing {config['desc']}...")
            
            if config.get("ssl"):
                # Use SMTP_SSL for port 465
                server = smtplib.SMTP_SSL(SMTP_SERVER, config["port"], timeout=10)
            else:
                server = smtplib.SMTP(SMTP_SERVER, config["port"], timeout=10)
                
            server.ehlo()
            
            if config.get("tls"):
                if server.has_extn('STARTTLS'):
                    server.starttls()
                    server.ehlo()
                else:
                    print("   ⚠️  STARTTLS not supported")
            
            print(f"   ✅ Connection successful on port {config['port']}!")
            
            # Try auth if this is the working port
            try:
                server.login(SMTP_USER, SMTP_PASSWORD)
                print(f"   ✅ Authentication successful!")
                server.quit()
                print(f"\n✅ Use port {config['port']} with TLS={config.get('tls', False)}\n")
                return config["port"], config.get("tls", False), config.get("ssl", False)
            except smtplib.SMTPAuthenticationError:
                print("   ⚠️  Auth failed, but connection works (may be open relay)")
                server.quit()
                return config["port"], config.get("tls", False), config.get("ssl", False)
                
        except Exception as e:
            print(f"   ❌ Failed: {type(e).__name__}: {e}")
    
    print("\n❌ All connection attempts failed\n")
    return None, None, None


# ── Usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # First, test connectivity
    working_port, use_tls, use_ssl = test_smtp_connection()
    
    if working_port:
        print(f"Found working configuration: Port {working_port}")
        print(f"Update SMTP_PORT to {working_port} in your config\n")
        
        # Try sending a test email
        response = input("Do you want to send a test email? (y/n): ")
        if response.lower() == 'y':
            send_email(
                to="rafik.atallah@Andalusiagroup.net",
                subject="QA mail",
                body_text="testing for qa agents.",
                review_link="https://your-internal-system.com/review?token=abc123"
            )
    else:
        print("⚠️  No working SMTP configuration found.")
        print("Possible issues:")
        print("  1. Firewall blocking outbound SMTP connections")
        print("  2. VPN required to access internal mail server")
        print("  3. SMTP server hostname is incorrect")
        print("  4. SMTP service is down")
        print("\nTry: ping mail.andalusiagroup.net")