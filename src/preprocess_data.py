"""
Data Preprocessing Script for GECKO Study

This script processes raw GECKO cholecystectomy data to create encoded patient and
hospital-level datasets ready for machine learning model training. It handles:
- Feature extraction from multi-value columns
- Categorical encoding (ordinal, binary, one-hot)
- Missing value imputation
- Geographic metadata mapping

Input: Raw GECKO CSV file (gecko.csv)
Output: Processed pickle files (df_encoded.pkl, df_site_encoded.pkl, mapping files)

Author: GECKO Study Team
"""

import argparse
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration and Constants
# =============================================================================

PATIENT_COLUMNS = [
    'patient_record_id', 'town', 'hospital', 'age_years', 'gender', 'asa', 'bmi', 'frailty',
    'comorb_n', 'pt_diabetes', 'pt_hypertension', 'pt_livdis', 'pt_tumour', 'pt_copd',
    'pt_pud', 'pt_comorbid_ckd', 'hist_ac', 'admission_prior', 'pre_img_us', 'pre_img_ct',
    'pre_img_mrcp', 'pre_img_ercp', 'pre_img_eus', 'pre_img_hida', 'pre_img_finding',
    'pre_img_finding_cbd', 'pre_symp_adm_day', 'pre_diag_dec_day', 'pre_dec_op_day',
    'pre_urgency', 'pre_urgency_ele_list', 'pre_indication', 'pre_indication_tg',
    'pre_indication_atlanta', 'op_anaes.grouped', 'op_anaes_local', 'op_anaes_regional',
    'op_anaes_inhaled', 'op_abx_indication', 'op_operator', 'op_operator_n',
    'op_operator_cons', 'op_operator_trainee_op', 'op_operator_trainee_cons', 'op_approach',
    'op_approach_open_reason', 'op_approach_conv_reason', 'op_approach_mis_gas',
    'op_approach_mis_reuse', 'op_nassar', 'op_cvs_elements', 'op_cvs_to',
    'op_performed_tcs', 'op_performed_stc', 'op_drain_yn', 'op_biliary_var',
    'op_img', 'op_img_desc_select', 'op_img_stone_mx', 'op_cbd_explore',
    'op_cbd_explore_close', 'op_contam', 'op_reuse_gown', 'op_comp', 'op_reuse_drape',
    'comps_chole_2l', 'comps_any_2l', 'comps_major_2l', 'bdi_yn'
]

INDEX_COLUMNS = ['patient_record_id', 'town', 'hospital']
NUMERICAL_COLUMNS = ['age_years', 'admission_prior', 'pre_symp_adm_day',
                     'pre_img_finding_cbd', 'pre_diag_dec_day', 'pre_dec_op_day', 'comorb_n']

SITE_FEATURES = [
    'town', 'hospital', 'hosp_type', 'hosp_type_2l', 'hosp_fund', 'hosp_fund_2l',
    'hosp_bed', 'hosp_itu_yn', 'hosp_itu_bed', 'hosp_hpb', 'hosp_hpb_oncall',
    'hosp_hpb_path', 'hosp_hpb_region', 'hosp_mis_yn', 'hosp_mis_type', 'hosp_mis_image',
    'service_chole_n', 'service_cons_n', 'service_cons_lap_n', 'service_cons_type',
    'service_type', 'service_eme_n', 'service_eme_theatre', 'day_unit', 'service_fu',
    'diag_us_yn', 'diag_ct_yn', 'diag_mrcp_yn', 'diag_eus_yn', 'diag_hida_yn',
    'diag_ostomy_yn', 'diag_ostomy_oncall', 'diag_ostomy_region', 'diag_ercp',
    'diag_ioc', 'diag_lu', 'diag_icg', 'diag_hist', 'train_perform', 'train_n',
    'train_grade', 'train_sim_yn', 'train_sim_type', 'tain_chole', 'train_bdi',
    'green_lap', 'green_surginst', 'green_drape', 'green_gown', 'green_scrubcap',
    'green_recycle', 'green_recycle_waste', 'green_ga', 'wb', 'iso2'
]

SITE_INDEX = ['town', 'hospital', 'iso2']
SITE_NUMERICAL = ['hosp_bed', 'hosp_itu_bed', 'service_chole_n', 'service_cons_n',
                  'service_cons_lap_n', 'service_eme_n', 'train_n']


# ISO2 to continent mapping for geographic analysis
ISO2_CONTINENT_MAP = {
    'US': 'North America', 'AU': 'Oceania', 'IT': 'Europe', 'YE': 'Asia', 'AE': 'Asia',
    'AL': 'Europe', 'AR': 'South America', 'AT': 'Europe', 'AW': 'North America',
    'AZ': 'Asia', 'BA': 'Europe', 'BD': 'Asia', 'BE': 'Europe', 'BG': 'Europe',
    'BI': 'Africa', 'BJ': 'Africa', 'BR': 'South America', 'CA': 'North America',
    'CH': 'Europe', 'CN': 'Asia', 'CO': 'South America', 'CY': 'Asia', 'CZ': 'Europe',
    'DE': 'Europe', 'DO': 'North America', 'DZ': 'Africa', 'EG': 'Africa', 'ES': 'Europe',
    'ET': 'Africa', 'FR': 'Europe', 'GB': 'Europe', 'GE': 'Asia', 'GH': 'Africa',
    'GR': 'Europe', 'GT': 'North America', 'HR': 'Europe', 'HU': 'Europe', 'ID': 'Asia',
    'IE': 'Europe', 'IL': 'Asia', 'IN': 'Asia', 'IQ': 'Asia', 'IR': 'Asia',
    'JM': 'North America', 'JO': 'Asia', 'JP': 'Asia', 'KE': 'Africa', 'KH': 'Asia',
    'KW': 'Asia', 'KZ': 'Asia', 'LB': 'Asia', 'LK': 'Asia', 'LT': 'Europe', 'LU': 'Europe',
    'LV': 'Europe', 'LY': 'Africa', 'MA': 'Africa', 'MK': 'Europe', 'MN': 'Asia',
    'MT': 'Europe', 'MX': 'North America', 'MY': 'Asia', 'NG': 'Africa', 'NO': 'Europe',
    'NZ': 'Oceania', 'OM': 'Asia', 'PA': 'North America', 'PE': 'South America',
    'PH': 'Asia', 'PK': 'Asia', 'PL': 'Europe', 'PS': 'Asia', 'PT': 'Europe',
    'PY': 'South America', 'QA': 'Asia', 'RO': 'Europe', 'RS': 'Europe', 'RU': 'Europe',
    'RW': 'Africa', 'SA': 'Asia', 'SD': 'Africa', 'SE': 'Europe', 'SG': 'Asia',
    'SN': 'Africa', 'SO': 'Africa', 'SV': 'North America', 'SY': 'Asia', 'TH': 'Asia',
    'TN': 'Africa', 'TR': 'Asia', 'TW': 'Asia', 'UA': 'Europe', 'UG': 'Africa',
    'UY': 'South America', 'VE': 'South America', 'ZA': 'Africa', 'EC': 'South America',
    'BY': 'Europe', 'NL': 'Europe', 'SI': 'Europe', 'BF': 'Africa', 'TZ': 'Africa',
    'HK': 'Asia', 'MO': 'Asia', 'GA': 'Africa', 'CM': 'Africa', 'IS': 'Europe', 'VN': 'Asia'
}

# ISO2 to ISO3 country code mapping
ISO2_TO_ISO3_MAP = {
    'US': 'USA', 'AU': 'AUS', 'IT': 'ITA', 'YE': 'YEM', 'AE': 'ARE', 'AL': 'ALB',
    'AR': 'ARG', 'AT': 'AUT', 'AW': 'ABW', 'AZ': 'AZE', 'BA': 'BIH', 'BD': 'BGD',
    'BE': 'BEL', 'BG': 'BGR', 'BI': 'BDI', 'BJ': 'BEN', 'BR': 'BRA', 'CA': 'CAN',
    'CH': 'CHE', 'CN': 'CHN', 'CO': 'COL', 'CY': 'CYP', 'CZ': 'CZE', 'DE': 'DEU',
    'DO': 'DOM', 'DZ': 'DZA', 'EG': 'EGY', 'ES': 'ESP', 'ET': 'ETH', 'FR': 'FRA',
    'GB': 'GBR', 'GE': 'GEO', 'GH': 'GHA', 'GR': 'GRC', 'GT': 'GTM', 'HR': 'HRV',
    'HU': 'HUN', 'ID': 'IDN', 'IE': 'IRL', 'IL': 'ISR', 'IN': 'IND', 'IQ': 'IRQ',
    'IR': 'IRN', 'JM': 'JAM', 'JO': 'JOR', 'JP': 'JPN', 'KE': 'KEN', 'KH': 'KHM',
    'KW': 'KWT', 'KZ': 'KAZ', 'LB': 'LBN', 'LK': 'LKA', 'LT': 'LTU', 'LU': 'LUX',
    'LV': 'LVA', 'LY': 'LBY', 'MA': 'MAR', 'MK': 'MKD', 'MN': 'MNG', 'MT': 'MLT',
    'MX': 'MEX', 'MY': 'MYS', 'NG': 'NGA', 'NO': 'NOR', 'NZ': 'NZL', 'OM': 'OMN',
    'PA': 'PAN', 'PE': 'PER', 'PH': 'PHL', 'PK': 'PAK', 'PL': 'POL', 'PS': 'PSE',
    'PT': 'PRT', 'PY': 'PRY', 'QA': 'QAT', 'RO': 'ROU', 'RS': 'SRB', 'RU': 'RUS',
    'RW': 'RWA', 'SA': 'SAU', 'SD': 'SDN', 'SE': 'SWE', 'SG': 'SGP', 'SN': 'SEN',
    'SO': 'SOM', 'SV': 'SLV', 'SY': 'SYR', 'TH': 'THA', 'TN': 'TUN', 'TR': 'TUR',
    'TW': 'TWN', 'UA': 'UKR', 'UG': 'UGA', 'UY': 'URY', 'VE': 'VEN', 'ZA': 'ZAF',
    'EC': 'ECU', 'BY': 'BLR', 'NL': 'NLD', 'SI': 'SVN', 'BF': 'BFA', 'TZ': 'TZA',
    'HK': 'HKG', 'MO': 'MAC', 'GA': 'GAB', 'CM': 'CMR', 'IS': 'ISL', 'VN': 'VNM'
}


# =============================================================================
# Patient Data Processing Functions
# =============================================================================

def extract_complication_features(df: pd.DataFrame, col_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for each complication type from the op_comp column.

    Args:
        df: DataFrame containing op_comp column
        col_list: List of column names to update

    Returns:
        Tuple of (updated DataFrame, updated column list)
    """
    complications = ['Bile spilt', 'Stones Spilt', 'Bleeding',
                     'Major vascular injury', 'Bowel injury']

    for complication in complications:
        col_name = f'has_{complication.lower().replace(" ", "_")}'
        df[col_name] = df['op_comp'].str.contains(complication, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['op_comp'])
    col_list.remove('op_comp')

    logging.info(f"Extracted {len(complications)} complication features")
    return df, col_list


def extract_imaging_findings(df: pd.DataFrame, col_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for each imaging finding from pre_img_finding column.

    Args:
        df: DataFrame containing pre_img_finding column
        col_list: List of column names to update

    Returns:
        Tuple of (updated DataFrame, updated column list)
    """
    findings = ['Gallstones', 'Thick-walled Gallbladder', 'Pericholecystic fluid',
                'CBD stones', 'Dilated CBD', 'Polyp(s)']

    for finding in findings:
        col_name = f'has_{finding.lower().replace(" ", "_").replace("(", "").replace(")", "")}'
        df[col_name] = df['pre_img_finding'].str.contains(finding, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['pre_img_finding'])
    col_list.remove('pre_img_finding')

    logging.info(f"Extracted {len(findings)} imaging finding features")
    return df, col_list


def extract_imaging_techniques(df: pd.DataFrame, col_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for intraoperative imaging techniques from op_img column.

    Args:
        df: DataFrame containing op_img column
        col_list: List of column names to update

    Returns:
        Tuple of (updated DataFrame, updated column list)
    """
    techniques_map = {
        'Intraoperative cholangiogram (IOC)': 'ioc',
        'Incisionless fluorescent cholangiography': 'fluorescent_cholangiography',
        'Intraoperative ERCP': 'intraop_ercp',
        'Laparoscopic ultrasound': 'lap_ultrasound'
    }

    for technique, short_name in techniques_map.items():
        col_name = f'has_{short_name}'
        df[col_name] = df['op_img'].str.contains(technique, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['op_img'])
    col_list.remove('op_img')

    logging.info(f"Extracted {len(techniques_map)} imaging technique features")
    return df, col_list


def extract_anesthesia_features(df: pd.DataFrame, col_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for anesthesia types from multiple columns.

    Args:
        df: DataFrame containing anesthesia columns
        col_list: List of column names to update

    Returns:
        Tuple of (updated DataFrame, updated column list)
    """
    # Local anesthesia types
    local_types = ['Subcutaneous', 'Intraperitoneal']
    for anesthesia_type in local_types:
        col_name = f'has_{anesthesia_type.lower()}'
        df[col_name] = df['op_anaes_local'].str.contains(anesthesia_type, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['op_anaes_local'])
    col_list.remove('op_anaes_local')

    # General anesthesia types
    anesthesia_types = ['General Inhaled', 'TIVA', 'Local', 'Sedation',
                        'Regional', 'Other combination']
    for anesthesia in anesthesia_types:
        col_name = f'has_{anesthesia.lower().replace(" ", "_")}'
        df[col_name] = df['op_anaes.grouped'].str.contains(anesthesia, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['op_anaes.grouped'])
    col_list.remove('op_anaes.grouped')

    logging.info(f"Extracted {len(local_types) + len(anesthesia_types)} anesthesia features")
    return df, col_list


def extract_cvs_elements(df: pd.DataFrame, col_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Critical View of Safety (CVS) element features from op_cvs_elements column.

    Args:
        df: DataFrame containing op_cvs_elements column
        col_list: List of column names to update

    Returns:
        Tuple of (updated DataFrame, updated column list)
    """
    elements_map = {
        'Clearing fat and fibrous tissue from the hepatocystic triangle': 'hepatocystic_triangle_cleared',
        'The lower third of the gallbladder being cleared from the cystic plate': 'lower_gb_cleared_from_plate',
        'Only two structures are attached to the gallbladder': 'two_structures_attached'
    }

    for element, short_name in elements_map.items():
        col_name = f'has_{short_name}'
        df[col_name] = df['op_cvs_elements'].str.contains(element, regex=False, na=False).astype(int)
        col_list.append(col_name)

    df = df.drop(columns=['op_cvs_elements'])
    col_list.remove('op_cvs_elements')

    logging.info(f"Extracted {len(elements_map)} CVS element features")
    return df, col_list


def standardize_binary_imaging_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize preoperative imaging columns to binary Yes/No format.

    Args:
        df: DataFrame with imaging columns

    Returns:
        DataFrame with standardized imaging columns
    """
    imaging_cols = ['pre_img_us', 'pre_img_ct', 'pre_img_mrcp',
                    'pre_img_ercp', 'pre_img_eus', 'pre_img_hida']

    for col in imaging_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 'Yes' if x == 'Yes' else 'No')

    logging.info(f"Standardized {len(imaging_cols)} imaging columns to Yes/No")
    return df


def encode_patient_features(df: pd.DataFrame, col_list: List[str],
                           index_list: List[str], numerical_list: List[str]) -> pd.DataFrame:
    """
    Encode patient features using ordinal, binary, and one-hot encoding strategies.

    Args:
        df: DataFrame to encode
        col_list: List of all columns to process
        index_list: List of index columns (not to encode)
        numerical_list: List of numerical columns (not to encode)

    Returns:
        Encoded DataFrame
    """
    # Clean data: replace 'Unknown' with NaN and handle special patterns
    cols_to_process = [col for col in col_list if col in df.columns
                      and col not in index_list and col not in numerical_list]

    for col in cols_to_process:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('Unknown', np.nan)
            df[col] = df[col].replace(regex=r'^Other\s*\{.*\}', value='Other')
            df[col] = df[col].replace(regex=r'^Yes\s*\{.*\}', value='Yes')

    # Extract frailty score number
    if 'frailty' in cols_to_process:
        df['frailty'] = df['frailty'].str.extract(r'^(\d)', expand=False)
        df['frailty'] = pd.to_numeric(df['frailty'], errors='coerce')

    # Define ordinal mappings
    ordinal_mappings = {
        'asa': {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5},
        'bmi': {'<18.5': 0, '18.5-24.9': 1, '25.0-29.9': 2, '30.0-34.9': 3,
                '35.0-39.9': 4, '40+': 5},
        'pt_comorbid_ckd': {'Stage I': 1, 'Stage II': 2, 'Stage IIIa': 3,
                           'Stage IIIb': 4, 'Stage IV': 5, 'Stage V': 6},
        'pre_indication_tg': {'I': 1, 'II': 2, 'III': 3},
        'pre_indication_atlanta': {'Mild': 1, 'Moderate': 2, 'Severe': 3},
        'op_operator_n': {'0-50': 0, '51-100': 1, '101-200': 2, '>200': 3},
        'op_nassar': {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5},
        'op_contam': {'Clean': 1, 'Clean-Contaminated': 2, 'Contaminated': 3, 'Dirty': 4}
    }

    # Apply ordinal mappings
    ordinal_cols_found = []
    for col, mapping in ordinal_mappings.items():
        if col in cols_to_process:
            df[col] = df[col].map(mapping)
            ordinal_cols_found.append(col)

    if 'frailty' in cols_to_process and 'frailty' not in ordinal_mappings:
        ordinal_cols_found.append('frailty')

    logging.info(f"Applied ordinal encoding to {len(ordinal_cols_found)} columns")

    # Define binary Yes/No columns
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = [
        'pt_diabetes', 'pt_hypertension', 'pt_livdis', 'pt_tumour', 'pt_copd',
        'pt_pud', 'hist_ac', 'pre_img_us', 'pre_img_ct', 'pre_img_mrcp',
        'pre_img_ercp', 'pre_img_eus', 'pre_img_hida', 'pre_urgency_ele_list',
        'op_operator_trainee_op', 'op_operator_trainee_cons', 'op_approach_mis_gas',
        'op_approach_mis_reuse', 'op_cvs_to', 'op_drain_yn', 'op_biliary_var',
        'op_reuse_gown', 'op_reuse_drape', 'comps_chole_2l', 'comps_any_2l',
        'comps_major_2l', 'bdi_yn'
    ]
    binary_cols = [col for col in binary_cols if col in cols_to_process]

    # Apply binary mappings
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    logging.info(f"Applied binary encoding to {len(binary_cols)} columns")

    # Identify columns for one-hot encoding
    one_hot_cols = [col for col in cols_to_process
                   if col not in ordinal_cols_found and col not in binary_cols
                   and df[col].dtype in ['object', 'category']]

    # Apply one-hot encoding
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols,
                           prefix_sep='_', dummy_na=True, drop_first=False)
        logging.info(f"Applied one-hot encoding to {len(one_hot_cols)} columns")

    # Convert any remaining boolean columns to int
    boolean_cols = df.select_dtypes(include='bool').columns
    if not boolean_cols.empty:
        for col in boolean_cols:
            df[col] = df[col].astype(int)

    return df


def impute_patient_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in patient data using median or zero strategies.

    Args:
        df: DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    # Columns for median imputation (numerical counts, ordinal features)
    cols_fill_median = [
        'pre_symp_adm_day', 'pre_dec_op_day', 'admission_prior', 'comorb_n',
        'pre_diag_dec_day', 'asa', 'bmi', 'pt_comorbid_ckd', 'pre_indication_tg',
        'pre_indication_atlanta', 'op_operator_n', 'op_nassar', 'op_contam', 'frailty'
    ]

    # Columns for zero imputation (binary flags)
    cols_fill_zero = [
        'pre_img_finding_cbd', 'pt_diabetes', 'pt_hypertension', 'pt_livdis',
        'pt_tumour', 'pt_copd', 'pt_pud', 'hist_ac', 'pre_urgency_ele_list',
        'op_operator_trainee_op', 'op_operator_trainee_cons', 'op_approach_mis_gas',
        'op_approach_mis_reuse', 'op_cvs_to', 'op_drain_yn', 'op_biliary_var',
        'op_reuse_gown', 'op_reuse_drape'
    ]

    # Filter to existing columns
    cols_fill_median = [col for col in cols_fill_median if col in df.columns]
    cols_fill_zero = [col for col in cols_fill_zero if col in df.columns]

    # Apply median imputation
    for col in cols_fill_median:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Apply zero imputation
    for col in cols_fill_zero:
        if df[col].isnull().any():
            df[col].fillna(0, inplace=True)

    # Fill missing town values
    df['town'].fillna('ahnor', inplace=True)

    remaining_nans = df.isnull().sum().sum()
    logging.info(f"Patient data imputation complete. Remaining NaNs: {remaining_nans}")

    return df


def process_patient_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to process patient-level data through all preprocessing steps.

    Args:
        data: Raw GECKO data

    Returns:
        Processed and encoded patient DataFrame
    """
    logging.info("Starting patient data processing...")

    col_list = PATIENT_COLUMNS.copy()
    filtered_data = data[col_list].copy()

    # Extract features from multi-value columns
    filtered_data, col_list = extract_complication_features(filtered_data, col_list)
    filtered_data, col_list = extract_imaging_findings(filtered_data, col_list)
    filtered_data, col_list = extract_imaging_techniques(filtered_data, col_list)
    filtered_data, col_list = extract_anesthesia_features(filtered_data, col_list)
    filtered_data, col_list = extract_cvs_elements(filtered_data, col_list)

    # Standardize imaging columns
    filtered_data = standardize_binary_imaging_columns(filtered_data)

    # Update numerical list with binary features
    numerical_list = NUMERICAL_COLUMNS.copy()
    binary_01_cols = [col for col in filtered_data.columns if col.startswith('has_')]
    numerical_list.extend([col for col in binary_01_cols if col in col_list])
    numerical_list = list(set(numerical_list))

    # Encode features
    df_encoded = encode_patient_features(filtered_data, col_list,
                                        INDEX_COLUMNS, numerical_list)

    # Impute missing values
    df_encoded = impute_patient_missing_values(df_encoded)

    logging.info(f"Patient data processing complete. Shape: {df_encoded.shape}")
    return df_encoded


# =============================================================================
# Hospital/Site Data Processing Functions
# =============================================================================

def extract_specialty_features(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for consultant specialties.

    Args:
        df: Site DataFrame
        feature_list: List of feature names

    Returns:
        Tuple of (updated DataFrame, updated feature list)
    """
    specialties = ['General', 'Upper GI', 'HPB', 'Colorectal', 'Breast', 'Other']

    for specialty in specialties:
        col_name = f'specialty_{specialty.lower().replace(" ", "_")}'
        df[col_name] = df['service_cons_type'].str.contains(specialty, regex=False, na=False).astype(int)
        feature_list.append(col_name)

    df = df.drop(columns=['service_cons_type'])
    feature_list.remove('service_cons_type')

    logging.info(f"Extracted {len(specialties)} specialty features")
    return df, feature_list


def extract_training_grade_features(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for training grades.

    Args:
        df: Site DataFrame
        feature_list: List of feature names

    Returns:
        Tuple of (updated DataFrame, updated feature list)
    """
    grades = ['Post-training fellow', 'Senior trainee', 'Junior trainee',
              'Non-trainees or doctors']

    for grade in grades:
        col_name = f'has_{grade.lower().replace(" ", "_").replace("-", "_")}'
        df[col_name] = df['train_grade'].str.contains(grade, regex=False, na=False).astype(int)
        feature_list.append(col_name)

    df = df.drop(columns=['train_grade'])
    feature_list.remove('train_grade')

    logging.info(f"Extracted {len(grades)} training grade features")
    return df, feature_list


def extract_simulation_features(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for simulation training types.

    Args:
        df: Site DataFrame
        feature_list: List of feature names

    Returns:
        Tuple of (updated DataFrame, updated feature list)
    """
    sim_types = ['Box trainer', 'IT simulation model', 'Animal model']

    for sim_type in sim_types:
        clean_name = sim_type.strip()
        col_name = f'has_{clean_name.lower().replace(" ", "_")}'
        df[col_name] = df['train_sim_type'].str.contains(sim_type, regex=False, na=False).astype(int)
        feature_list.append(col_name)

    df = df.drop(columns=['train_sim_type'])
    feature_list.remove('train_sim_type')

    logging.info(f"Extracted {len(sim_types)} simulation training features")
    return df, feature_list


def extract_bdi_training_features(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract binary features for BDI (Bile Duct Injury) training levels.

    Args:
        df: Site DataFrame
        feature_list: List of feature names

    Returns:
        Tuple of (updated DataFrame, updated feature list)
    """
    bdi_types_map = {
        'Yes (Local hospital)': 'local_bdi',
        'Yes (Regional)': 'regional_bdi',
        'Yes (National)': 'national_bdi'
    }

    for bdi_type, short_name in bdi_types_map.items():
        col_name = f'has_{short_name}'
        df[col_name] = df['train_bdi'].str.contains(bdi_type, regex=False, na=False).astype(int)
        feature_list.append(col_name)

    df = df.drop(columns=['train_bdi'])
    feature_list.remove('train_bdi')

    logging.info(f"Extracted {len(bdi_types_map)} BDI training features")
    return df, feature_list


def encode_site_features(df: pd.DataFrame, feature_list: List[str],
                        index_list: List[str], numerical_list: List[str]) -> pd.DataFrame:
    """
    Encode hospital/site features using ordinal, binary, and one-hot encoding.

    Args:
        df: Site DataFrame
        feature_list: List of all features
        index_list: Index columns
        numerical_list: Numerical columns

    Returns:
        Encoded site DataFrame
    """
    cols_to_process = [col for col in feature_list if col in df.columns
                      and col not in index_list and col not in numerical_list]

    # Define ordinal mappings for site features
    ordinal_mappings = {
        'hosp_hpb_oncall': {
            'Every day (24 hour)': 3, 'Weekdays only (24 hour)': 2,
            'Every day (daytime 0800 - 1700)': 1, 'Weekdays only (daytime 0800 - 1700)': 0
        },
        'service_eme_theatre': {
            'Yes (Everyday)': 4, 'Yes (More than once every 2 weeks)': 3,
            'Yes (Once a week)': 2, 'Yes (Once every 2 week)': 1, 'No': 0
        },
        'service_fu': {'Yes (Routinely)': 2, 'Yes (Selectively)': 1, 'No': 0},
        'diag_us_yn': {'Available on-site': 1, 'Available off-site': 0},
        'diag_ct_yn': {'Available on-site': 2, 'Available off-site': 1, 'Not available': 0},
        'diag_mrcp_yn': {'Available on-site': 2, 'Available off-site': 1, 'Not available': 0},
        'diag_eus_yn': {'Available on-site': 2, 'Available off-site': 1, 'Not available': 0},
        'diag_hida_yn': {'Available on-site': 2, 'Available off-site': 1, 'Not available': 0},
        'diag_ostomy_oncall': {
            'Yes (Everyday)': 4, 'Yes (More than once every 2 weeks)': 3,
            'Yes (Once a week)': 2, 'Yes (Once every 2 week)': 1, 'No': 0
        },
        'diag_ercp': {
            'Yes (Everyday)': 4, 'Yes (More than once every 2 weeks)': 3,
            'Yes (Once a week)': 2, 'Yes (Once every 2 week)': 1, 'No': 0
        },
        'diag_ioc': {
            'Routine use': 3, 'Selective use with good supply': 2,
            'Selective use with limited supply': 1, 'Not available': 0
        },
        'diag_lu': {
            'Routine use': 3, 'Selective use with good supply': 2,
            'Selective use with limited supply': 1, 'Not available': 0
        },
        'diag_icg': {
            'Routine use': 3, 'Selective use with good supply': 2,
            'Selective use with limited supply': 1, 'Not available': 0
        },
        'diag_hist': {
            'Yes (Routinely)': 2, 'Yes (Selectively)': 1,
            'No (Not sent for histology)': 0, 'No (No access to histology)': 0
        },
        'green_lap': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not used but available)': 1, 'No (Not available)': 0
        },
        'green_surginst': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not used but available)': 1, 'No (Not available)': 0
        },
        'green_drape': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not used but available)': 1, 'No (Not available)': 0
        },
        'green_gown': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not used but available)': 1, 'No (Not available)': 0
        },
        'green_scrubcap': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not used but available)': 1, 'No (Not available)': 0
        },
        'green_recycle': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Not recycled but recycling service available)': 1,
            'No (No recycling service available)': 0
        },
        'green_ga': {
            'Yes (Always)': 3, 'Yes (Sometimes)': 2,
            'No (Anaesthetic gases used but IV anaesthesia available)': 1,
            'No (No IV anaesthesia available)': 0
        },
        'wb': {'High income': 2, 'Upper middle income': 1, 'Lower middle / Low income': 0}
    }

    # Apply ordinal mappings
    ordinal_cols_found = []
    for col, mapping in ordinal_mappings.items():
        if col in cols_to_process:
            df[col] = df[col].map(mapping)
            ordinal_cols_found.append(col)

    logging.info(f"Applied ordinal encoding to {len(ordinal_cols_found)} site columns")

    # Define binary Yes/No columns
    binary_map = {'Yes': 1, 'No': 0}
    yes_no_cols = [
        'hosp_type_2l', 'hosp_fund_2l', 'hosp_itu_yn', 'hosp_hpb', 'hosp_hpb_path',
        'hosp_mis_yn', 'day_unit', 'diag_ostomy_yn', 'train_perform',
        'train_sim_yn', 'tain_chole', 'green_recycle_waste'
    ]
    yes_no_cols = [col for col in yes_no_cols if col in cols_to_process]

    # Apply binary mappings
    for col in yes_no_cols:
        df[col] = df[col].map(binary_map)

    logging.info(f"Applied binary encoding to {len(yes_no_cols)} site columns")

    # Identify columns for one-hot encoding
    one_hot_cols = [col for col in cols_to_process
                   if col not in ordinal_cols_found and col not in yes_no_cols
                   and df[col].dtype in ['object', 'category']]

    # Apply one-hot encoding
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols,
                           prefix_sep='_', dummy_na=True, drop_first=False)
        logging.info(f"Applied one-hot encoding to {len(one_hot_cols)} site columns")

    # Convert boolean columns to int
    boolean_cols = df.select_dtypes(include='bool').columns
    if not boolean_cols.empty:
        for col in boolean_cols:
            df[col] = df[col].astype(int)

    return df


def impute_site_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in hospital/site data.

    Args:
        df: Site DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    # Columns for median imputation
    cols_fill_median = [
        'hosp_itu_bed', 'service_cons_n', 'service_cons_lap_n', 'service_chole_n',
        'train_n', 'service_eme_n', 'hosp_hpb_oncall', 'service_eme_theatre',
        'diag_eus_yn', 'diag_hida_yn', 'diag_ostomy_oncall', 'diag_ioc', 'diag_lu',
        'green_surginst', 'green_ga'
    ]

    # Columns for zero imputation
    cols_fill_zero = [
        'hosp_hpb_path', 'diag_ostomy_yn', 'tain_chole', 'green_recycle_waste'
    ]

    # Filter to existing columns
    cols_fill_median = [col for col in cols_fill_median if col in df.columns]
    cols_fill_zero = [col for col in cols_fill_zero if col in df.columns]

    # Apply median imputation
    for col in cols_fill_median:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Apply zero imputation
    for col in cols_fill_zero:
        if df[col].isnull().any():
            df[col].fillna(0, inplace=True)

    # Fill missing location values
    df['town'].fillna('ahnor', inplace=True)

    # Fix specific missing ISO2 codes
    df.loc[(df['town'] == 'wal') & (df['hospital'] == 'welwits'), 'iso2'] = 'NA'
    df.loc[(df['town'] == 'wind') & (df['hospital'] == 'wincwah'), 'iso2'] = 'WD'

    remaining_nans = df.isnull().sum().sum()
    logging.info(f"Site data imputation complete. Remaining NaNs: {remaining_nans}")

    return df


def process_site_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to process hospital/site-level data through all preprocessing steps.

    Args:
        data: Raw GECKO data

    Returns:
        Processed and encoded site DataFrame
    """
    logging.info("Starting site data processing...")

    # Select site features and remove duplicates
    site_df = data[SITE_FEATURES].copy()
    site_df = site_df.drop_duplicates()

    # Verify no duplicates by town and hospital
    duplicates = site_df.duplicated(subset=['town', 'hospital']).sum()
    if duplicates > 0:
        logging.warning(f"Found {duplicates} duplicate town-hospital combinations")

    feature_list = SITE_FEATURES.copy()

    # Extract features from multi-value columns
    site_df, feature_list = extract_specialty_features(site_df, feature_list)
    site_df, feature_list = extract_training_grade_features(site_df, feature_list)
    site_df, feature_list = extract_simulation_features(site_df, feature_list)
    site_df, feature_list = extract_bdi_training_features(site_df, feature_list)

    # Update numerical list with binary features
    site_numerical = SITE_NUMERICAL.copy()
    binary_site_cols = [col for col in site_df.columns
                       if col.startswith('specialty_') or col.startswith('has_')]
    site_numerical.extend([col for col in binary_site_cols if col in feature_list])
    site_numerical = list(set(site_numerical))

    # Encode features
    df_site_encoded = encode_site_features(site_df, feature_list,
                                          SITE_INDEX, site_numerical)

    # Impute missing values
    df_site_encoded = impute_site_missing_values(df_site_encoded)

    logging.info(f"Site data processing complete. Shape: {df_site_encoded.shape}")
    return df_site_encoded


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def preprocess_gecko_data(input_path: str, output_dir: str) -> Dict[str, str]:
    """
    Main preprocessing pipeline for GECKO data.

    Args:
        input_path: Path to raw gecko.csv file
        output_dir: Directory to save processed files

    Returns:
        Dictionary with paths to output files
    """
    # Load raw data
    logging.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path, sep=',')
    logging.info(f"Loaded {len(data)} records")

    # Process patient data
    df_encoded = process_patient_data(data)

    # Process site data
    df_site_encoded = process_site_data(data)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    patient_path = os.path.join(output_dir, 'df_encoded.pkl')
    site_path = os.path.join(output_dir, 'df_site_encoded.pkl')

    df_encoded.to_pickle(patient_path)
    df_site_encoded.to_pickle(site_path)

    logging.info(f"Saved patient data to {patient_path}")
    logging.info(f"Saved site data to {site_path}")

    # Save geographic mappings
    iso_map_path = os.path.join(output_dir, 'iso_map.pkl')
    iso_ver_map_path = os.path.join(output_dir, 'iso_ver_map.pkl')

    with open(iso_map_path, 'wb') as f:
        pickle.dump(ISO2_CONTINENT_MAP, f)

    with open(iso_ver_map_path, 'wb') as f:
        pickle.dump(ISO2_TO_ISO3_MAP, f)

    logging.info(f"Saved ISO mappings to {output_dir}")

    # Log summary statistics
    logging.info("\n=== Data Summary ===")
    logging.info(f"Patient records: {len(df_encoded)}")
    logging.info(f"Patient features: {df_encoded.shape[1]}")
    logging.info(f"Unique hospitals: {df_site_encoded['hospital'].nunique()}")
    logging.info(f"Site features: {df_site_encoded.shape[1]}")
    logging.info(f"Complication rate (comps_chole_2l): {df_encoded['comps_chole_2l'].sum()}/{len(df_encoded)} "
                f"({df_encoded['comps_chole_2l'].mean():.1%})")

    return {
        'patient_data': patient_path,
        'site_data': site_path,
        'iso_map': iso_map_path,
        'iso_ver_map': iso_ver_map_path
    }


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Preprocess GECKO cholecystectomy data for ML model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python preprocess_data.py --input /path/to/gecko.csv --output ./processed

  # Specify custom data directory
  python preprocess_data.py --input D:/Data/gecko.csv --output ./data/processed

  # Enable debug logging
  python preprocess_data.py --input gecko.csv --output ./processed --log-level DEBUG
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input GECKO CSV file (gecko.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./processed',
        help='Output directory for processed pickle files (default: ./processed)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Validate input file
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return 1

    try:
        # Run preprocessing pipeline
        output_paths = preprocess_gecko_data(args.input, args.output)

        logging.info("\n=== Preprocessing Complete ===")
        logging.info("Output files:")
        for key, path in output_paths.items():
            logging.info(f"  {key}: {path}")

        return 0

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
