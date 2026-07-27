# Hybrid AI Email Classification System 

An intelligent, two-stage cybersecurity pipeline that detects and classifies emails as **SAFE**, **SPAM**, or **PHISHING**. 

This project bridges the gap between classical Machine Learning and modern Generative AI. It uses a blazing-fast Logistic Regression model for initial triage, and seamlessly escalates uncertain or suspicious emails to a large language model (Llama 3 via Groq) for deep, human-like contextual analysis.

## 🧠 Architecture Overview

The system uses a **Triage Architecture** to balance API costs, processing speed, and maximum security:

1. **Step 1: Fast ML Model (Local)**
   * Incoming emails are vectorized using TF-IDF and passed through a pre-trained Logistic Regression model (`Logistic_Regression_model.pkl`).
   * The model calculates a confidence score (Safe vs. Suspicious).
2. **Step 2: The Gatekeeper (Logic)**
   * If the ML model is highly confident (≥ 99.5%) that the email is SAFE, the pipeline stops and accepts the prediction, saving time and API costs.
   * If the email is flagged as SUSPICIOUS, or if the confidence score is low, it is escalated to Step 3.
3. **Step 3: Deep LLM Analysis (Cloud)**
   * The email is sent to `llama-3.3-70b-versatile` via the Groq API, guided by a massive, highly detailed system prompt containing few-shot examples.
   * The LLM reads the context, identifies typosquatting, urgency triggers, and malicious links, and returns a definitive JSON classification along with the specific threat signals detected.

## ⚙️ Features
* **Automated Test Suite:** Instantly run a batch of pre-configured phishing and spam scenarios to verify pipeline accuracy.
* **Interactive CLI Mode:** Paste any email subject, body, and links directly into the terminal for real-time analysis.
* **Live Inbox Monitor:** Connect to a real Gmail inbox using IMAP. The script runs in "Demo Mode," taking a snapshot of the backlog and only triggering the AI pipeline when brand new emails arrive.

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Run the following command in your terminal to install the required libraries. 
*(Note: `scikit-learn==1.8.0` is strictly required to match the environment used to train the `.pkl` model).*
```bash
pip install groq python-dotenv joblib numpy scikit-learn==1.8.0
2. Get an "App Password" (For Gmail)
If you are using Gmail for the Live Monitor, Google will block standard password logins from Python scripts for security reasons. You must generate a special "App Password".

Go to your Google Account settings -> Security.

Make sure 2-Step Verification is turned on.

Search for App Passwords in the search bar.

Create a new App Password (name it "Python Classifier" or similar).

It will give you a 16-character password (e.g., abcd efgh ijkl mnop). Copy this without spaces.

3. Update your .env file
Create a file named .env in the root of your project directory and add your API keys and email credentials. Never commit this file to version control.

Code snippet
# Groq API Key
GROQ_API_KEY=YOUR_GROQ_API

# IMAP Email Credentials
IMAP_SERVER=imap.gmail.com
EMAIL_ACCOUNT=your.email@gmail.com
EMAIL_PASSWORD=abcdefghijklmnop
🚀 Usage Guide
1. Run the Automated Test Suite
To verify that the ML model and LLM are communicating correctly, run the built-in test suite:

Bash
python test_pipeline.py
2. Run Interactive Mode (Manual Testing)
Test a specific email by pasting its details directly into the terminal:

Bash
python test_pipeline.py -i
3. Run the Live Inbox Monitor (Demo Mode)
To actively monitor your real email inbox for incoming threats:

Bash
python live_monitor.py
The monitor will safely ignore your existing unread emails to protect your API limits and will only analyze brand new emails that arrive while the script is running.

📁 File Structure
pipeline.py: The main orchestrator that routes data between the ML model and the LLM.

ml_classifier.py: Loads the .pkl file and handles the TF-IDF / Logistic Regression logic.

llm_analyzer.py: Contains the system prompt and handles the Groq API requests.

test_pipeline.py: The CLI interface for automated and interactive testing.

live_monitor.py: The IMAP bridge that connects the pipeline to a live email inbox.

Logistic_Regression_model.pkl: The serialized, pre-trained machine learning model.

Note:
Software & Dependencies
Python: Version 3.10 or higher installed on your machine.

Python Libraries: You need to install the specific packages that make the ML model and API bridge work. You can get them all at once by running this in your terminal:

Bash
pip install groq python-dotenv joblib numpy scikit-learn==1.8.0
