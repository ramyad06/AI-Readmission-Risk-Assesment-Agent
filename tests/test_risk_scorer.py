import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.risk_scorer import compute_risk_score


class TestLowRiskPatient:
    def test_low_risk_score_and_level(self):
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
        assert result["score"] <= 3, f"Expected score <= 3, got {result['score']}"
        assert result["contributing_factors"] == [], "Expected no contributing factors"

    def test_low_risk_breakdown_sums_to_score(self):
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
    def test_medium_risk_no_followup_and_comorbidities(self):
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
        assert result["risk_level"] == "Medium", f"Expected Medium, got {result['risk_level']}"
        assert 4 <= result["score"] <= 6, f"Expected score 4-6, got {result['score']}"

    def test_medium_risk_factors_listed(self):
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
        assert "Comorbidity count >= 3" in result["contributing_factors"]


class TestHighRiskPatient:
    def test_high_risk_maximum_score(self):
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
        patient = {
            "age": 76,
            "prior_admissions_6mo": 2,
            "length_of_stay_days": 8,
            "follow_up_scheduled": "No",
            "comorbidity_count": 0,
            "medication_count": 3,
            "discharge_destination": "SNF",
        }
        result = compute_risk_score(patient)
        assert result["score"] == 7, f"Expected score 7, got {result['score']}"
        assert result["risk_level"] == "High"


class TestEdgeCases:
    def test_young_patient_high_medication_count(self):
        patient = {
            "age": 28,
            "prior_admissions_6mo": 0,
            "length_of_stay_days": 3,
            "follow_up_scheduled": "Yes",
            "comorbidity_count": 0,
            "medication_count": 11,
            "discharge_destination": "Home",
        }
        result = compute_risk_score(patient)
        assert result["score"] == 2
        assert result["risk_level"] == "Low"
        assert "Medication count >= 8" in result["contributing_factors"]

    def test_elderly_perfect_followup_low_risk(self):
        patient = {
            "age": 80,
            "prior_admissions_6mo": 0,
            "length_of_stay_days": 4,
            "follow_up_scheduled": "Yes",
            "comorbidity_count": 1,
            "medication_count": 3,
            "discharge_destination": "SNF",
        }
        result = compute_risk_score(patient)
        assert result["score"] == 2
        assert result["risk_level"] == "Low"


class TestMissingFieldHandling:
    def test_empty_patient_dict_returns_valid_result(self):
        result = compute_risk_score({})
        assert isinstance(result["score"], int)
        assert result["risk_level"] in ("Low", "Medium", "High")
        assert isinstance(result["contributing_factors"], list)
        assert isinstance(result["score_breakdown"], dict)

    def test_non_numeric_age_treated_as_zero(self):
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
