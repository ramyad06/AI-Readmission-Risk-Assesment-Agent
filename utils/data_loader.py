import os
import pandas as pd
from typing import Any, Dict, Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_CSV_PATH = os.path.join(_DATA_DIR, "patients.csv")


def load_all_patients() -> pd.DataFrame:
    if not os.path.exists(_CSV_PATH):
        raise FileNotFoundError(
            f"Patient dataset not found at: {_CSV_PATH}\n"
            "Ensure data/patients.csv exists relative to the project root."
        )
    df = pd.read_csv(_CSV_PATH, dtype={"patient_id": str})
    return df


def list_patient_ids() -> list[str]:
    df = load_all_patients()
    return sorted(df["patient_id"].tolist())


def load_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    df = load_all_patients()
    patient_id = str(patient_id).strip().upper()
    match = df[df["patient_id"].str.upper() == patient_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()