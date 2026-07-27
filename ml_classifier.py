"""
ml_classifier.py
----------------
Step 1 of the Hybrid Email Classification Pipeline.
Loads the pre-trained Logistic Regression + TF-IDF pipeline and classifies
incoming emails. Returns prediction and confidence score.

Model classes:
  0 = SAFE (not spam/phishing)
  1 = SPAM or PHISHING (suspicious — further analysis needed)
"""

import joblib
import numpy as np
import os

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Logistic_Regression_model.pkl")

# Confidence threshold: if ML confidence >= this, accept ML result directly.
# If below this, escalate to LLM for deep analysis.
CONFIDENCE_THRESHOLD = 0.995
#0.80

def load_model():
    """Load and return the trained pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please ensure Logistic_Regression_model.pkl is in the same directory."
        )
    return joblib.load(MODEL_PATH)


# Load model once at module import
_model = load_model()


def classify_email(subject: str, body: str) -> dict:
    """
    Classify an email using the ML model.
    
    Parameters:
    -----------
    subject : Email subject line
    body    : Email body text
    
    Returns:
    --------
    dict with keys:
      - raw_label      : int (0=SAFE, 1=SUSPICIOUS)
      - label_name     : str ("SAFE" or "SUSPICIOUS")
      - confidence     : float (confidence for the predicted class)
      - safe_prob      : float (probability of being safe)
      - suspicious_prob: float (probability of being suspicious)
      - escalate_to_llm: bool (True if confidence < threshold)
    """
    # Combine subject + body for the model (TF-IDF processes raw text)
    combined_text = f"{subject} {body}"
    
    # Get prediction and probabilities
    probabilities = _model.predict_proba([combined_text])[0]
    raw_label = int(_model.predict([combined_text])[0])
    
    safe_prob = float(probabilities[0])
    suspicious_prob = float(probabilities[1])
    
    # Confidence = probability of the predicted class
    confidence = suspicious_prob if raw_label == 1 else safe_prob
    
    label_name = "SUSPICIOUS" if raw_label == 1 else "SAFE"
    
 # ALWAYS escalate to LLM if the ML model thinks it is suspicious, 
    # so the LLM can accurately classify it as either SPAM or PHISHING.
    escalate = (confidence < CONFIDENCE_THRESHOLD) or (raw_label == 1)
    
    return {
        "raw_label": raw_label,
        "label_name": label_name,
        "confidence": round(confidence, 4),
        "safe_prob": round(safe_prob, 4),
        "suspicious_prob": round(suspicious_prob, 4),
        "escalate_to_llm": escalate
    }