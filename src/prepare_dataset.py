"""
Dataset preparation for ContextSurg model.

This script prepares patient and hospital data for training the ContextSurg model.
"""

import os
import pickle
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_dir: str, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load patient and hospital data from pickle files.

    Args:
        data_dir: Directory containing data files
        target: Target variable name

    Returns:
        Tuple of (patient_df, hospital_df)
    """
    patient_path = os.path.join(data_dir, 'df_encoded.pkl')
    hospital_path = os.path.join(data_dir, 'df_site_encoded.pkl')

    if not os.path.exists(patient_path) or not os.path.exists(hospital_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    with open(patient_path, 'rb') as f:
        patient_df = pickle.load(f)

    with open(hospital_path, 'rb') as f:
        hospital_df = pickle.load(f)

    logging.info(f"Loaded {len(patient_df)} patients from {len(hospital_df.drop_duplicates(['town', 'hospital']))} hospitals")

    return patient_df, hospital_df


def prepare_dataset(
    data_dir: str = './data/processed',
    target: str = 'comps_chole_2l',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict:
    """
    Prepare dataset for ContextSurg training.

    Args:
        data_dir: Directory containing processed data files
        target: Target variable to predict
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing prepared datasets
    """
    # Load data
    patient_df, hospital_df = load_data(data_dir, target)

    # Merge patient and hospital data
    merged_df = pd.merge(
        patient_df,
        hospital_df,
        on=['town', 'hospital'],
        how='left'
    )

    # Define feature columns
    ID_COLS = ['patient_record_id', 'town', 'hospital', 'iso2']
    TARGET_COL = target

    # Get all columns except IDs and target
    all_cols = set(merged_df.columns)
    feature_cols = all_cols - set(ID_COLS) - {TARGET_COL}

    # Separate patient and hospital features
    # Hospital features are those from hospital_df
    hospital_feature_cols = [col for col in hospital_df.columns
                            if col not in ID_COLS]

    patient_feature_cols = [col for col in feature_cols
                           if col not in hospital_feature_cols]

    # Extract features and target
    X_patient = merged_df[patient_feature_cols].fillna(merged_df[patient_feature_cols].median())
    X_hospital = merged_df[hospital_feature_cols].fillna(merged_df[hospital_feature_cols].median())
    y = merged_df[TARGET_COL]
    location = merged_df['wb']  # World Bank income group as location

    # Create location mapping
    unique_locations = location.dropna().unique()
    location_map = {loc: idx for idx, loc in enumerate(sorted(unique_locations))}
    location_idx = location.map(location_map).fillna(0).astype(int)

    # Split data
    # First split: train+val vs test
    X_patient_tv, X_patient_test, \
    X_hospital_tv, X_hospital_test, \
    y_tv, y_test, \
    location_tv, location_test = train_test_split(
        X_patient, X_hospital, y, location_idx,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_patient_train, X_patient_val, \
    X_hospital_train, X_hospital_val, \
    y_train, y_val, \
    location_train, location_val = train_test_split(
        X_patient_tv, X_hospital_tv, y_tv, location_tv,
        test_size=val_size,
        random_state=random_state,
        stratify=y_tv
    )

    logging.info(f"Split sizes - Train: {len(X_patient_train)}, "
                f"Val: {len(X_patient_val)}, Test: {len(X_patient_test)}")

    # Get unique hospital features for global context
    unique_hospital_train = X_hospital_train.drop_duplicates()

    # Scale features
    patient_scaler = StandardScaler()
    hospital_scaler = StandardScaler()

    X_patient_train_scaled = patient_scaler.fit_transform(X_patient_train)
    X_patient_val_scaled = patient_scaler.transform(X_patient_val)
    X_patient_test_scaled = patient_scaler.transform(X_patient_test)

    X_hospital_train_scaled = hospital_scaler.fit_transform(X_hospital_train)
    X_hospital_val_scaled = hospital_scaler.transform(X_hospital_val)
    X_hospital_test_scaled = hospital_scaler.transform(X_hospital_test)

    unique_hospital_train_scaled = hospital_scaler.transform(unique_hospital_train)

    # Convert to tensors
    dataset = {
        'X_train': torch.FloatTensor(X_patient_train_scaled),
        'X_val': torch.FloatTensor(X_patient_val_scaled),
        'X_test': torch.FloatTensor(X_patient_test_scaled),
        'y_train': torch.FloatTensor(y_train.values).unsqueeze(1),
        'y_val': torch.FloatTensor(y_val.values).unsqueeze(1),
        'y_test': torch.FloatTensor(y_test.values).unsqueeze(1),
        'hospital_train': torch.FloatTensor(X_hospital_train_scaled),
        'hospital_val': torch.FloatTensor(X_hospital_val_scaled),
        'hospital_test': torch.FloatTensor(X_hospital_test_scaled),
        'location_train': torch.LongTensor(location_train.values),
        'location_val': torch.LongTensor(location_val.values),
        'location_test': torch.LongTensor(location_test.values),
        'all_train_hospital_features': torch.FloatTensor(unique_hospital_train_scaled),
        'patient_feature_cols': patient_feature_cols,
        'hospital_feature_cols': hospital_feature_cols,
        'location_map': location_map,
        'patient_scaler': patient_scaler,
        'hospital_scaler': hospital_scaler,
        'target': target
    }

    return dataset


def main():
    """Example usage of dataset preparation."""
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset for ContextSurg')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory containing data files')
    parser.add_argument('--target', type=str, default='comps_chole_2l',
                       help='Target variable')
    parser.add_argument('--output', type=str, default='./data/dataset.pkl',
                       help='Output file path')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataset = prepare_dataset(
        data_dir=args.data_dir,
        target=args.target
    )

    # Save prepared dataset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)

    logging.info(f"Dataset saved to {args.output}")


if __name__ == '__main__':
    main()
