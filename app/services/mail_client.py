import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Config ─────────────────────────────────────────────────────────
SMTP_SERVER   = "mail.andalusiagroup.net"    # Your internal Exchange SMTP host
SMTP_PORT     = 587                     # 25 (relay) or 587 (TLS) or 465 (SSL)
SENDER_EMAIL  = "rafik.atallah@andalusiagroup.net"
SMTP_USER     = "andalusia\\rafik.atallah"      # Or just "username@andalusia.com"
SMTP_PASSWORD = "rafik123"        # Leave empty if using anonymous relay


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
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()          # Remove if using port 25 anonymous relay
            server.login(SMTP_USER, SMTP_PASSWORD)  # Remove if anonymous relay
            server.sendmail(SENDER_EMAIL, to, msg.as_string())
            print(f"✅ Email sent to {to}")

    except Exception as e:
        print(f"❌ Failed: {e}")


# ── Usage ───────────────────────────────────────────────────────────
send_email(
    to="Yasser.Hamed@Andalusiagroup.net",
    subject="QA mail",
    body_text="testing for qa agents.",
    review_link="https://your-internal-system.com/review?token=abc123"
)