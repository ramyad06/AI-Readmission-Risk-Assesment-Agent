"""
test_risk_scorer.py — Unit tests for the rule-based risk scoring engine.

Covers: Low Risk, Medium Risk, High Risk, edge cases, and missing field handling.
Run with: python -m pytest tests/test_risk_scorer.py -v
"""

import sys
import os
import json

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.risk_scorer import compute_risk_score


class TestLowRiskPatient:
    """Test scoring for a patient who should be classified as Low Risk."""

    def test_low_risk_score_and_level(self):
        """Young patient, no prior admissions, short stay, follow-up booked."""
        patient = {
            "age": 35,
            "prior_admissions_6mo": 0,
            "length_of_stay_days": 3,
            "follow_up_scheduled": "Yes",
            "comorbidity_count": 0,
            "medication_count": 2,
            "discharge_destination": "Home+Services",
        }
        result = compute_risk_score(patient)
        assert result["risk_level"] == "Low", f"Expected Low, got {result['risk_level']}"
        assert result["score"] <= 3, f"Expected score ≤ 3, got {result['score']}"
        assert result["contributing_factors"] == [], "Expected no contributing factors"

    def test_low_risk_breakdown_sums_to_score(self):
        """Verify the breakdown values sum to the total score."""
        patient = {
            "age": 40,
            "prior_admissions_6mo": 1,
            "length_of_stay_days": 5,
            "follow_up_scheduled": "Yes",
            "comorbidity_count": 1,
            "medication_count": 3,
            "discharge_destination": "SNF",
        }
        result = compute_risk_score(patient)
        breakdown_total = sum(result["score_breakdown"].values())
        assert breakdown_total == result["score"], (
            f"Breakdown sum {breakdown_total} != score {result['score']}"
        )


class TestMediumRiskPatient:
    """Test scoring for patients who should land in Medium Risk (4–6)."""

    def test_medium_risk_no_followup_and_comorbidities(self):
        """Patient with no follow-up, prior admissions, and comorbidities → Medium Risk.

        Score breakdown:
          +2  No follow-up scheduled
          +2  Prior admissions ≥ 2
          +1  Comorbidity count ≥ 3
          ─────────────────────────
          = 5  → Medium Risk
        """
        patient = {
            "age": 60,
            "prior_admissions_6mo": 2,   # triggers +2
            "length_of_stay_days": 6,
            "follow_up_scheduled": "No",  # triggers +2
            "comorbidity_count": 3,       # triggers +1
            "medication_count": 5,
            "discharge_destination": "Rehab",
        }
        result = compute_risk_score(patient)
        assert result["risk_level"] == "Medium", f"Expected Medium, got {result['risk_level']}"
        assert 4 <= result["score"] <= 6, f"Expected score 4–6, got {result['score']}"

    def test_medium_risk_factors_listed(self):
        """The contributing_factors list must name all triggered rules."""
        patient = {
            "age": 60,
            "prior_admissions_6mo": 2,
            "length_of_stay_days": 6,
            "follow_up_scheduled": "No",
            "comorbidity_count": 3,
            "medication_count": 5,
            "discharge_destination": "Rehab",
        }
        result = compute_risk_score(patient)
        assert "No follow-up scheduled" in result["contributing_factors"]
        assert "Comorbidity count ≥ 3" in result["contributing_factors"]


class TestHighRiskPatient:
    """Test scoring for patients who should be classified as High Risk (7–10)."""

    def test_high_risk_maximum_score(self):
        """Elderly patient with all risk factors triggered → High Risk, score 10."""
        patient = {
            "age": 85,
            "prior_admissions_6mo": 5,
            "length_of_stay_days": 18,
            "follow_up_scheduled": "No",
            "comorbidity_count": 5,
            "medication_count": 14,
            "discharge_destination": "Home",
        }
        result = compute_risk_score(patient)
        assert result["risk_level"] == "High", f"Expected High, got {result['risk_level']}"
        assert result["score"] == 10, f"Expected score 10, got {result['score']}"
        assert len(result["contributing_factors"]) == 7

    def test_high_risk_boundary_score_7(self):
        """Test the boundary condition: score exactly 7 should be High Risk."""
        patient = {
            "age": 76,          # +2
            "prior_admissions_6mo": 2,  # +2
            "length_of_stay_days": 8,   # +1
            "follow_up_scheduled": "No",  # +2
            "comorbidity_count": 0,     # +0
            "medication_count": 3,      # +0
            "discharge_destination": "SNF",  # +0
        }
        result = compute_risk_score(patient)
        assert result["score"] == 7, f"Expected score 7, got {result['score']}"
        assert result["risk_level"] == "High"


class TestEdgeCases:
    """Test edge cases and unusual patient profiles."""

    def test_young_patient_high_medication_count(self):
        """Edge case: 28-year-old with 11 medications but no other risk factors."""
        patient = {
            "age": 28,           # +0
            "prior_admissions_6mo": 0,   # +0
            "length_of_stay_days": 3,    # +0
            "follow_up_scheduled": "Yes",  # +0
            "comorbidity_count": 0,      # +0
            "medication_count": 11,      # +1
            "discharge_destination": "Home",  # +1
        }
        result = compute_risk_score(patient)
        assert result["score"] == 2
        assert result["risk_level"] == "Low"
        assert "Medication count ≥ 8" in result["contributing_factors"]

    def test_elderly_perfect_followup_low_risk(self):
        """Edge case: 80-year-old with perfect follow-up scores Low Risk (age only)."""
        patient = {
            "age": 80,           # +2
            "prior_admissions_6mo": 0,   # +0
            "length_of_stay_days": 4,    # +0
            "follow_up_scheduled": "Yes",  # +0
            "comorbidity_count": 1,      # +0
            "medication_count": 3,       # +0
            "discharge_destination": "SNF",  # +0
        }
        result = compute_risk_score(patient)
        assert result["score"] == 2
        assert result["risk_level"] == "Low"


class TestMissingFieldHandling:
    """Test that the scorer handles missing or malformed fields gracefully."""

    def test_empty_patient_dict_returns_valid_result(self):
        """Missing all fields should produce a score (no crash), likely Medium due to no follow-up."""
        result = compute_risk_score({})
        assert isinstance(result["score"], int)
        assert result["risk_level"] in ("Low", "Medium", "High")
        assert isinstance(result["contributing_factors"], list)
        assert isinstance(result["score_breakdown"], dict)

    def test_non_numeric_age_treated_as_zero(self):
        """A non-numeric age should not crash; it should be treated as 0."""
        patient = {
            "age": "unknown",
            "prior_admissions_6mo": 0,
            "length_of_stay_days": 3,
            "follow_up_scheduled": "Yes",
            "comorbidity_count": 0,
            "medication_count": 2,
            "discharge_destination": "Home+Services",
        }
        result = compute_risk_score(patient)
        assert result["score_breakdown"]["age"] == 0

    def test_follow_up_case_insensitive(self):
        """follow_up_scheduled should be case-insensitive ('YES' treated as 'Yes')."""
        patient = {
            "age": 50,
            "prior_admissions_6mo": 0,
            "length_of_stay_days": 3,
            "follow_up_scheduled": "YES",
            "comorbidity_count": 0,
            "medication_count": 2,
            "discharge_destination": "Home+Services",
        }
        result = compute_risk_score(patient)
        assert result["score_breakdown"]["no_follow_up"] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
