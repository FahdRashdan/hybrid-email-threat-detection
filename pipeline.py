"""
pipeline.py
-----------
Hybrid Email Classification Pipeline — Main Orchestrator.

Flow:
  Email Input
      ↓
  Step 1: ML Model (Fast Classification)
      ↓
  Confidence Decision
      ├─ High Confidence (≥ 0.80) → Accept ML result directly
      └─ Low Confidence  (< 0.80) → Escalate to LLM
                                         ↓
                              Step 2: LLM Deep Analysis
                                         ↓
  Final Output: PHISHING | SPAM | SAFE + details
"""

from ml_classifier import classify_email
from llm_analyzer import analyze_email_with_llm


def run_pipeline(
    subject: str,
    body: str,
    sender: str = "",
    attachment_text: str = "",
    links: list = None
) -> dict:
    """
    Run the full hybrid classification pipeline on an email.
    
    Parameters:
    -----------
    subject         : Email subject line
    body            : Email body text  
    sender          : Sender's email address (optional but recommended)
    attachment_text : Text extracted from PDF/DOCX attachments (optional)
    links           : List of URLs found in the email (optional)
    
    Returns:
    --------
    dict with full pipeline results
    """
    print("\n" + "="*60)
    print("  HYBRID EMAIL CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"  Subject : {subject[:80]}{'...' if len(subject) > 80 else ''}")
    print(f"  From    : {sender or 'Not provided'}")
    print("="*60)

    # ─────────────────────────────────────────────
    # STEP 1: ML Model
    # ─────────────────────────────────────────────
    print("\n[STEP 1] Running ML Model (Fast Classification)...")
    ml_result = classify_email(subject, body)
    
    print(f"  ML Label      : {ml_result['label_name']}")
    print(f"  ML Confidence : {ml_result['confidence']:.2%}")
    print(f"  Safe Prob     : {ml_result['safe_prob']:.2%}")
    print(f"  Suspicious    : {ml_result['suspicious_prob']:.2%}")

    # ─────────────────────────────────────────────
    # STEP 2: Confidence Decision
    # ─────────────────────────────────────────────
    if not ml_result["escalate_to_llm"]:
        print(f"\n[DECISION] High confidence ({ml_result['confidence']:.2%} ≥ 0.80)")
        print("           → Accepting ML classification directly.")
        
        # Map ML label to final output
        if ml_result["raw_label"] == 0:
            final_classification = "SAFE"
        else:
            # ML says suspicious but doesn't distinguish PHISHING vs SPAM
            # Default to SPAM for high-confidence ML detections at this stage
            final_classification = "SPAM"
        
        return {
            "final_classification": final_classification,
            "decided_by": "ML_MODEL",
            "ml_result": ml_result,
            "llm_result": None,
            "confidence": ml_result["confidence"],
            "recommend_manual_review": False,
            "signals_detected": [
                f"ML_HIGH_CONFIDENCE: {ml_result['confidence']:.2%} confidence for {ml_result['label_name']}",
                f"ML_SAFE_PROB: {ml_result['safe_prob']:.2%}",
                f"ML_SUSPICIOUS_PROB: {ml_result['suspicious_prob']:.2%}"
            ],
            "reasoning": (
                f"The ML model classified this email as {ml_result['label_name']} with "
                f"{ml_result['confidence']:.2%} confidence, which exceeds the threshold of 80%. "
                "LLM escalation was not required."
            )
        }
    
    # ─────────────────────────────────────────────
    # STEP 2: Escalate to LLM
    # ─────────────────────────────────────────────
    print(f"\n[DECISION] Low confidence ({ml_result['confidence']:.2%} < 0.80)")
    print("           → Escalating to LLM for deep analysis...")
    print("\n[STEP 2] Running LLM Analysis (Few-Shot Deep Analysis)...")
    
    ml_label_str = ml_result["label_name"]
    
    llm_result = analyze_email_with_llm(
        subject=subject,
        body=body,
        sender=sender,
        attachment_text=attachment_text,
        links=links or [],
        ml_prediction=ml_label_str,
        ml_confidence=ml_result["confidence"]
    )
    
    final_classification = llm_result.get("classification", "SAFE")
    
    print(f"  LLM Classification : {final_classification}")
    print(f"  LLM Confidence     : {llm_result.get('confidence', 0):.2%}")
    print(f"  Manual Review?     : {llm_result.get('recommend_manual_review', False)}")
    
    return {
        "final_classification": final_classification,
        "decided_by": "LLM_ANALYSIS",
        "ml_result": ml_result,
        "llm_result": llm_result,
        "confidence": llm_result.get("confidence", 0.5),
        "recommend_manual_review": llm_result.get("recommend_manual_review", False),
        "signals_detected": llm_result.get("signals_detected", []),
        "reasoning": llm_result.get("reasoning", "")
    }


def print_final_result(result: dict):
    """Pretty-print the final pipeline result."""
    classification = result["final_classification"]
    
    # Color codes for terminal
    colors = {
        "PHISHING": "\033[91m",  # Red
        "SPAM":     "\033[93m",  # Yellow
        "SAFE":     "\033[92m",  # Green
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    
    color = colors.get(classification, "")
    
    print("\n" + "="*60)
    print(f"  {BOLD}FINAL RESULT{RESET}")
    print("="*60)
    print(f"  Classification : {BOLD}{color}{classification}{RESET}")
    print(f"  Confidence     : {result['confidence']:.2%}")
    print(f"  Decided By     : {result['decided_by']}")
    print(f"  Manual Review  : {'⚠️  YES' if result['recommend_manual_review'] else 'No'}")
    
    if result["signals_detected"]:
        print(f"\n  Signals Detected:")
        for signal in result["signals_detected"]:
            print(f"    • {signal}")
    
    if result["reasoning"]:
        print(f"\n  Reasoning:")
        # Word-wrap the reasoning
        words = result["reasoning"].split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
    
    print("="*60 + "\n")