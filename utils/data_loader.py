"""
data_loader.py — Utility functions for loading patient records from CSV.

All data is synthetic and anonymised. No real patient information is used.
"""

import os
import pandas as pd
from typing import Optional

# Resolve the absolute path to the CSV regardless of CWD
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_CSV_PATH = os.path.join(_DATA_DIR, "patients.csv")


def load_all_patients() -> pd.DataFrame:
    """Load all patient records from the CSV file into a DataFrame.

    Returns:
        pd.DataFrame: All 40 synthetic patient records with original columns.

    Raises:
        FileNotFoundError: If patients.csv cannot be found at the expected path.
    """
    if not os.path.exists(_CSV_PATH):
        raise FileNotFoundError(
            f"Patient dataset not found at: {_CSV_PATH}\n"
            "Ensure data/patients.csv exists relative to the project root."
        )
    df = pd.read_csv(_CSV_PATH, dtype={"patient_id": str})
    return df


def list_patient_ids() -> list:
    """Return a sorted list of all patient_id values from the dataset.

    Returns:
        list[str]: Sorted list of patient IDs, e.g. ['P001', 'P002', ...].
    """
    df = load_all_patients()
    return sorted(df["patient_id"].tolist())


def load_patient_by_id(patient_id: str) -> Optional[dict]:
    """Retrieve a single patient record as a dictionary by their patient_id.

    Args:
        patient_id (str): The patient identifier, e.g. 'P007'.

    Returns:
        dict | None: Patient record as a flat dictionary, or None if not found.

    Example:
        >>> record = load_patient_by_id("P007")
        >>> record["age"]
        85
    """
    df = load_all_patients()
    patient_id = str(patient_id).strip().upper()
    match = df[df["patient_id"].str.upper() == patient_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()
