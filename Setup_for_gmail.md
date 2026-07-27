### Step 1: Get an "App Password" (For Gmail)
If you are using Gmail, Google will block standard password logins from Python scripts for security reasons. You must generate a special "App Password".

1. Go to your **Google Account** settings -> **Security**.
2. Make sure **2-Step Verification** is turned on.
3. Search for **App Passwords** in the search bar.
4. Create a new App Password (name it "Python Classifier" or similar).
5. It will give you a 16-character password (e.g., `abcd efgh ijkl mnop`). Copy this.

### Step 2: Update your `.env` file
Open your `.env` file and add your email credentials. Do not include spaces in the app password.

**`.env`**
```env
# Groq API Key
GROQ_API_KEY= YOUR_API_KEY

# IMAP Email Credentials
IMAP_SERVER=imap.gmail.com
EMAIL_ACCOUNT=your.email@gmail.com
EMAIL_PASSWORD=abcdefghijklmnop