import os
import time
import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv

# Import your existing pipeline functions!
from pipeline import run_pipeline, print_final_result

load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def decode_mime_words(s):
    """Decodes email subjects and senders properly."""
    if not s:
        return ""
    decoded_words = decode_header(s)
    text = ""
    for word, encoding in decoded_words:
        if isinstance(word, bytes):
            text += word.decode(encoding or "utf-8", errors="ignore")
        else:
            text += str(word)
    return text

def get_email_body(msg):
    """Extracts plain text body from the email payload."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Look for plain text parts that are not attachments
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except Exception:
            pass
    return body.strip()

def monitor_inbox():
    """Connects to IMAP, ignores the backlog, and checks for NEW emails in a loop."""
    print("\n" + "="*60)
    print(" 🛡️  LIVE INBOX MONITOR STARTED (DEMO MODE)")
    print(f" 📧 Listening to: {EMAIL_ACCOUNT}")
    print(" Press Ctrl+C to stop.")
    print("="*60 + "\n")

    try:
        # Connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select("inbox")
    except Exception as e:
        print(f"❌ Failed to connect to email server: {e}")
        print("Please check your .env file and ensure you are using an App Password.")
        return

    # --- INITIAL SNAPSHOT: IGNORE THE BACKLOG ---
    ignored_ids = set()
    status, messages = mail.search(None, "UNSEEN")
    if status == "OK" and messages[0]:
        ignored_ids = set(messages[0].split())
        print(f" [i] Ignoring {len(ignored_ids)} existing unread emails to protect Groq API limits.")
        print(" Waiting for brand new emails to arrive...\n")
    else:
        print(" [i] Inbox is clear. Waiting for brand new emails to arrive...\n")
    # --------------------------------------------

    while True:
        try:
            mail.select("inbox")
            # Search for all unread emails
            status, messages = mail.search(None, "UNSEEN")
            
            if status == "OK" and messages[0]:
                current_ids = messages[0].split()
                
                for e_id in current_ids:
                    # ONLY process if it's a completely new ID we haven't seen yet
                    if e_id not in ignored_ids:
                        print(f"\n[!] BRAND NEW email detected (ID: {e_id.decode()}). Fetching...")
                        
                        # Fetch the email data
                        res, msg_data = mail.fetch(e_id, "(RFC822)")
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_bytes(response_part[1])
                                
                                # Parse headers
                                subject = decode_mime_words(msg.get("Subject"))
                                sender = decode_mime_words(msg.get("From"))
                                body = get_email_body(msg)
                                
                                # RUN YOUR PIPELINE!
                                result = run_pipeline(
                                    subject=subject,
                                    body=body,
                                    sender=sender,
                                    attachment_text="", 
                                    links=[] 
                                )
                                
                                print_final_result(result)
                        
                        # Add to ignored list so we don't process it again in the next loop
                        ignored_ids.add(e_id)
            
            # Wait 10 seconds before checking again so we don't spam the server
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            break
        except Exception as e:
            print(f"\n⚠️ Error during email fetch: {e}")
            time.sleep(10)

    mail.logout()

if __name__ == "__main__":
    if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
        print("❌ Error: Missing EMAIL_ACCOUNT or EMAIL_PASSWORD in .env file.")
    else:
        monitor_inbox()