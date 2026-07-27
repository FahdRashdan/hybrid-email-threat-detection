"""
test_pipeline.py
----------------
Interactive test suite for the Hybrid Email Classification Pipeline.

Usage:
  python test_pipeline.py                  # Run all built-in test cases
  python test_pipeline.py --interactive    # Enter your own email to test
  python test_pipeline.py --test 1         # Run a specific test case (1-8)
"""

import sys
import argparse
from pipeline import run_pipeline, print_final_result

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": 1,
        "name": "Classic Bank Phishing",
        "expected": "PHISHING",
        "email": {
            "sender": "security@chase-verify-account.com",
            "subject": "URGENT: Your Chase Bank Account Has Been Suspended",
            "body": """Dear Valued Customer,

We have detected suspicious activity on your Chase Bank account and have
temporarily suspended your access for your protection.

To restore your account access immediately, you must verify your identity
within 24 hours by clicking the link below:

>> VERIFY YOUR ACCOUNT NOW: http://chase-account-verify.net/login

Failure to verify within 24 hours will result in permanent account closure
and funds being frozen pending investigation.

Chase Bank Security Department
© 2024 JPMorgan Chase & Co. All Rights Reserved.""",
            "links": ["http://chase-account-verify.net/login"]
        }
    },
    {
        "id": 2,
        "name": "Commercial Spam - Flash Sale",
        "expected": "SPAM",
        "email": {
            "sender": "newsletter@shopdeals24.com",
            "subject": "🔥 MEGA SALE: 70% OFF Everything — 48 Hours Only!",
            "body": """Hi Shopper!

Our biggest sale of the year is HERE! For the next 48 hours only,
get up to 70% off on ALL categories:

👗 Fashion — up to 70% off
📱 Electronics — up to 50% off
🏠 Home & Garden — up to 60% off

Use promo code: MEGA70 at checkout

Shop now at www.shopdeals24.com/mega-sale

You received this email because you subscribed to ShopDeals24 Newsletter.
Click here to unsubscribe | View Privacy Policy | Contact Support""",
            "links": ["www.shopdeals24.com/mega-sale"]
        }
    },
    {
        "id": 3,
        "name": "Legitimate Work Email",
        "expected": "SAFE",
        "email": {
            "sender": "manager@company.com",
            "subject": "Team Meeting Notes - Q4 Planning Session",
            "body": """Hi everyone,

Thanks for joining today's Q4 planning session. Here's a quick recap:

Key Decisions:
- Launch date confirmed for November 15th
- Budget allocation approved: $250K for marketing
- Ahmed will lead the product team for the new feature rollout

Action Items:
1. Sara: Send updated roadmap by Friday
2. Khaled: Coordinate with the design team this week
3. Fahd: Prepare the client presentation for next Monday

Please review the attached meeting notes and let me know if I missed anything.

Best,
Manager
Company Corp""",
            "links": []
        }
    },
    {
        "id": 4,
        "name": "Microsoft Spear Phishing (AI-Generated)",
        "expected": "PHISHING",
        "email": {
            "sender": "itsupport@microsofft-365.com",
            "subject": "Action Required: Your Microsoft 365 License Expires in 3 Days",
            "body": """Dear Employee,

Your Microsoft 365 Business Premium license is scheduled to expire in 3 days.
To avoid service interruption and loss of access to Outlook, Teams, and SharePoint,
please renew your license immediately through the IT portal.

License Information:
  - Account: your.email@company.com
  - License: Microsoft 365 Business Premium
  - Expiration: 3 days from today
  - Renewal Cost: $22.00/month

Renew Now: https://ms365-license-portal.net/renew

If you do not renew within 72 hours, your account will be deactivated and
all cloud data may be inaccessible.

IT Support Team
Microsoft Partner Services""",
            "links": ["https://ms365-license-portal.net/renew"]
        }
    },
    {
        "id": 5,
        "name": "Package Delivery Scam",
        "expected": "PHISHING",
        "email": {
            "sender": "noreply@fedx-delivery.com",
            "subject": "Your Package Cannot Be Delivered — Pay $2.99 Redelivery Fee",
            "body": """Dear Customer,

We attempted to deliver your package today but were unable to complete
delivery due to an unconfirmed address.

Package Details:
  Tracking: FDX-7743821-US
  Carrier: FedEx Express
  Status: Delivery Attempted — Address Unconfirmed

To reschedule your delivery, please pay the small redelivery fee of $2.99
and confirm your address using the secure link below:

>> CONFIRM ADDRESS & PAY FEE: http://fedx-delivery.com/redelivery/7743821

This must be completed within 48 hours or your package will be returned
to the sender.

FedEx Customer Support""",
            "links": ["http://fedx-delivery.com/redelivery/7743821"]
        }
    },
    {
        "id": 6,
        "name": "Prize / Lottery Phishing",
        "expected": "PHISHING",
        "email": {
            "sender": "rewards@amazon-prize-center.com",
            "subject": "Congratulations! You've Won a $1000 Amazon Gift Card",
            "body": """Dear Amazon Customer,

You have been selected as our monthly prize winner!

As a valued Amazon customer, you are entitled to receive a $1,000 Amazon
Gift Card. This is our way of thanking you for your continued loyalty.

To claim your prize, you need to:
1. Verify your Amazon account credentials
2. Complete a short survey (2 minutes)
3. Provide your shipping address for the gift card delivery

Claim Your Prize Here: http://amazon-prize-center.com/claim?ref=monthly

IMPORTANT: This offer expires in 6 hours. Only 10 prizes remaining.

Amazon Customer Loyalty Team""",
            "links": ["http://amazon-prize-center.com/claim?ref=monthly"]
        }
    },
    {
        "id": 7,
        "name": "Newsletter Subscription (Borderline Spam)",
        "expected": "SPAM",
        "email": {
            "sender": "updates@techweekly-digest.com",
            "subject": "This Week in Tech: AI Breakthroughs, New iPhone Leaks & More",
            "body": """Hello Tech Enthusiast,

Welcome to this week's TechWeekly Digest! Here's what happened in tech:

🤖 AI NEWS: OpenAI announces new model with improved reasoning
📱 APPLE: iPhone 17 leaks suggest major camera upgrade  
💻 MICROSOFT: Windows 12 release date reportedly confirmed
🔒 SECURITY: Critical vulnerability found in popular VPN software

Read the full stories at: www.techweekly-digest.com/this-week

Want early access to tomorrow's digest? Upgrade to Premium for $4.99/month.

You're receiving this because you signed up at TechWeekly.com
Unsubscribe | Manage Preferences | Privacy Policy""",
            "links": ["www.techweekly-digest.com/this-week"]
        }
    },
    {
        "id": 8,
        "name": "Internal IT Security Alert (Safe)",
        "expected": "SAFE",
        "email": {
            "sender": "it-security@company.com",
            "subject": "Scheduled System Maintenance — Saturday 2AM-4AM",
            "body": """Dear Team,

Please be aware of scheduled system maintenance this Saturday from 2:00 AM
to 4:00 AM (Cairo time). The following systems will be temporarily unavailable:

- Email server
- VPN access  
- Internal file server

No action is required from your side. Systems will restore automatically.
If you experience any issues after 4:00 AM, please contact IT support at
it-support@company.com or call extension 4440.

Thank you for your understanding.

IT Security Team
Company Corp""",
            "links": []
        }
    }
]


def run_single_test(test_case: dict, verbose: bool = True) -> bool:
    """Run a single test case. Returns True if prediction matches expected."""
    print(f"\n{'━'*60}")
    print(f"  TEST CASE #{test_case['id']}: {test_case['name']}")
    print(f"  Expected: {test_case['expected']}")
    print(f"{'━'*60}")
    
    email = test_case["email"]
    result = run_pipeline(
        subject=email["subject"],
        body=email["body"],
        sender=email.get("sender", ""),
        attachment_text=email.get("attachment_text", ""),
        links=email.get("links", [])
    )
    
    print_final_result(result)
    
    got = result["final_classification"]
    expected = test_case["expected"]
    
    if got == expected:
        print(f"  ✅ PASS — Got: {got} | Expected: {expected}\n")
        return True
    else:
        print(f"  ❌ FAIL — Got: {got} | Expected: {expected}\n")
        return False


def run_interactive():
    """Interactive mode: user enters their own email to test."""
    print("\n" + "="*60)
    print("  INTERACTIVE EMAIL CLASSIFIER TEST")
    print("="*60)
    print("Enter your email details below. Press Enter twice to submit.\n")
    
    sender = input("Sender email (press Enter to skip): ").strip()
    subject = input("Subject: ").strip()
    
    print("Body (type your text, then type 'END' on a new line when done):")
    body_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        body_lines.append(line)
    body = "\n".join(body_lines)
    
    links_input = input("\nLinks (comma-separated, or press Enter to skip): ").strip()
    links = [l.strip() for l in links_input.split(",") if l.strip()] if links_input else []
    
    attachment_text = ""
    has_attachment = input("\nDo you have attachment text to analyze? (y/n): ").strip().lower()
    if has_attachment == "y":
        print("Attachment text (type 'END' on a new line when done):")
        att_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            att_lines.append(line)
        attachment_text = "\n".join(att_lines)
    
    result = run_pipeline(
        subject=subject,
        body=body,
        sender=sender,
        attachment_text=attachment_text,
        links=links
    )
    print_final_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="Test the Hybrid Email Classification Pipeline"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter your own email interactively"
    )
    parser.add_argument(
        "--test", "-t",
        type=int,
        choices=range(1, len(TEST_CASES) + 1),
        help=f"Run a specific test case (1-{len(TEST_CASES)})"
    )
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive()
        return
    
    if args.test:
        test_case = next(t for t in TEST_CASES if t["id"] == args.test)
        run_single_test(test_case)
        return
    
    # Run ALL test cases
    print("\n" + "█"*60)
    print("  RUNNING ALL TEST CASES")
    print("█"*60)
    
    passed = 0
    failed = 0
    
    for test_case in TEST_CASES:
        success = run_single_test(test_case)
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "█"*60)
    print(f"  TEST SUMMARY: {passed}/{len(TEST_CASES)} PASSED")
    print(f"  ✅ Passed: {passed}  |  ❌ Failed: {failed}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()