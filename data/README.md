# Data Directory

This directory should contain your preprocessed data files for training and evaluation.

## Directory Structure

```
data/
├── processed/           # Preprocessed training data
│   ├── df_encoded.pkl
│   ├── df_site_encoded.pkl
│   └── iso_map.pkl
└── external_validation/ # External validation datasets (optional)
    └── <external_dataset>.pkl
```

## Data Format Requirements

### Training Data Files

#### 1. `df_encoded.pkl` - Patient-level Data

A pandas DataFrame where each row represents a patient. Required columns:

**Identifier Columns:**
- `patient_record_id` (str/int): Unique patient identifier
- `town` (str): Town/city name where patient was treated
- `hospital` (str): Hospital name
- `iso2` (str): ISO2 country code (e.g., 'US', 'GB', 'IN')

**Target Columns** (at least one required):
- `comps_chole_2l` (int): Binary outcome for cholecystectomy complications (0/1)
- `comps_major_2l` (int): Binary outcome for major complications (0/1)
- `comps_any_2l` (int): Binary outcome for any complications (0/1)

**Feature Columns:**
- Clinical features (e.g., age, BMI, comorbidities)
- Laboratory values
- Preoperative assessments
- Other patient characteristics

**Example:**
```python
import pandas as pd

df_patient = pd.DataFrame({
    'patient_record_id': [1, 2, 3, ...],
    'town': ['London', 'Paris', 'Berlin', ...],
    'hospital': ['Hospital A', 'Hospital B', ...],
    'iso2': ['GB', 'FR', 'DE', ...],
    'comps_chole_2l': [0, 1, 0, ...],
    'age': [45, 67, 52, ...],
    'bmi': [25.3, 31.2, 28.5, ...],
    # ... other features
})

df_patient.to_pickle('processed/df_encoded.pkl')
```

#### 2. `df_site_encoded.pkl` - Hospital-level Data

A pandas DataFrame where each row represents a unique hospital. Required columns:

**Identifier Columns:**
- `town` (str): Town/city name
- `hospital` (str): Hospital name
- `iso2` (str): ISO2 country code
- `wb` (int): World Bank income group (0=low, 1=middle, 2=high)

**Feature Columns:**
- Hospital resources (beds, ICU capacity, equipment)
- Hospital type (teaching, district, rural, etc.)
- Available services
- Training programs
- Geographic characteristics

**Example:**
```python
df_hospital = pd.DataFrame({
    'town': ['London', 'Paris', 'Berlin', ...],
    'hospital': ['Hospital A', 'Hospital B', ...],
    'iso2': ['GB', 'FR', 'DE', ...],
    'wb': [2, 2, 2, ...],  # World Bank income group
    'hosp_bed': [500, 300, 450, ...],  # Number of beds
    'hosp_itu_yn': [1, 1, 0, ...],  # Has ICU (yes/no)
    # ... other hospital features
})

df_hospital.to_pickle('processed/df_site_encoded.pkl')
```

#### 3. `iso_map.pkl` - ISO2 to Continent Mapping

A Python dictionary mapping ISO2 country codes to continent names:

```python
iso2continent = {
    'US': 'North America',
    'GB': 'Europe',
    'FR': 'Europe',
    'IN': 'Asia',
    'BR': 'South America',
    'ZA': 'Africa',
    'AU': 'Oceania',
    # ... more mappings
}

import pickle
with open('processed/iso_map.pkl', 'wb') as f:
    pickle.dump(iso2continent, f)
```

### Data Preprocessing

Before training, ensure your data is:

1. **Encoded**: Categorical variables should be one-hot encoded or label encoded
2. **Cleaned**: Missing values handled appropriately
3. **Normalized**: Consider normalizing/standardizing numerical features (done automatically by the training script)
4. **Validated**: Check for data quality issues, outliers, and inconsistencies
