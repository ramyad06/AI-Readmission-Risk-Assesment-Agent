"""
validate_agent.py — Batch validation of the risk agent against all 40 CSV patients.

Compares the rule-based predicted risk level against the readmitted_30d ground truth
and prints a simple accuracy / confusion summary.

Usage:
    python validate_agent.py

No Ollama connection required — uses direct deterministic scoring only.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from utils.data_loader import load_all_patients
from tools.risk_scorer import compute_risk_score


# Risk level → binary readmission prediction mapping
# High / Medium → predicted readmitted; Low → not readmitted
RISK_TO_BINARY = {"High": "Yes", "Medium": "Yes", "Low": "No"}


def run_validation() -> None:
    """Run the scoring engine against all 40 patients and print a confusion summary.

    Compares:
        - Predicted: derived from rule-based risk_level (High/Medium → readmitted)
        - Actual: readmitted_30d column (Yes / No)

    Prints a detailed per-patient table and an aggregate confusion matrix.
    """
    df = load_all_patients()
    results = []

    for _, row in df.iterrows():
        patient = row.to_dict()
        assessment = compute_risk_score(patient)
        risk_level = assessment["risk_level"]
        score = assessment["score"]
        predicted = RISK_TO_BINARY[risk_level]
        actual = str(row["readmitted_30d"]).strip()

        results.append({
            "patient_id": row["patient_id"],
            "age": row["age"],
            "primary_condition": row["primary_condition"],
            "risk_level": risk_level,
            "score": score,
            "predicted_readmit": predicted,
            "actual_readmit": actual,
            "correct": predicted == actual,
        })

    results_df = pd.DataFrame(results)

    # ── Per-patient output ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BATCH VALIDATION RESULTS — 40 SYNTHETIC PATIENTS")
    print("=" * 80)
    print(f"{'ID':<8} {'Age':<5} {'Condition':<16} {'Risk':<8} {'Score':<7} "
          f"{'Pred':<7} {'Actual':<8} {'✓/✗'}")
    print("-" * 80)

    for _, r in results_df.iterrows():
        tick = "✓" if r["correct"] else "✗"
        print(
            f"{r['patient_id']:<8} {r['age']:<5} {r['primary_condition']:<16} "
            f"{r['risk_level']:<8} {r['score']:<7} {r['predicted_readmit']:<7} "
            f"{r['actual_readmit']:<8} {tick}"
        )

    # ── Summary statistics ─────────────────────────────────────────────────────
    total = len(results_df)
    correct = results_df["correct"].sum()
    accuracy = correct / total * 100

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Total patients : {total}")
    print(f"  Correct        : {correct}")
    print(f"  Incorrect      : {total - correct}")
    print(f"  Accuracy       : {accuracy:.1f}%")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    tp = ((results_df["predicted_readmit"] == "Yes") & (results_df["actual_readmit"] == "Yes")).sum()
    tn = ((results_df["predicted_readmit"] == "No")  & (results_df["actual_readmit"] == "No")).sum()
    fp = ((results_df["predicted_readmit"] == "Yes") & (results_df["actual_readmit"] == "No")).sum()
    fn = ((results_df["predicted_readmit"] == "No")  & (results_df["actual_readmit"] == "Yes")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n  CONFUSION MATRIX (Readmission = Positive Class)")
    print(f"  {'':>20} | Predicted YES | Predicted NO")
    print(f"  {'-'*50}")
    print(f"  {'Actual YES (readmitted)':>20} | {tp:^13} | {fn:^12}")
    print(f"  {'Actual NO':>20} | {fp:^13} | {tn:^12}")
    print(f"\n  Precision : {precision:.2f}")
    print(f"  Recall    : {recall:.2f}")
    print(f"  F1 Score  : {f1:.2f}")

    # ── Risk distribution ──────────────────────────────────────────────────────
    dist = results_df["risk_level"].value_counts()
    print(f"\n  RISK DISTRIBUTION")
    for level in ["High", "Medium", "Low"]:
        count = dist.get(level, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"  {level:<8}: {count:>3} patients ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 80)
    print("  NOTE: This is a rule-based heuristic — not a trained ML model.")
    print("  Results are for demonstration and educational purposes only.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_validation()
