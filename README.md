# ContextSurg: Harnessing Context-Aware Deep Learning to Predict Postoperative Complications Across 108 Countries

This repository contains the official implementation of the ContextSurg model.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

The model expects preprocessed data in the following format:

### Training Data

Place your preprocessed data files in `data/processed/`:

- `df_encoded.pkl`: Patient-level features (pandas DataFrame)
- `df_site_encoded.pkl`: Hospital-level features (pandas DataFrame)
- `iso_map.pkl`: ISO2 to continent mapping (dict)

**Data Format:**

- **Patient data** (`df_encoded.pkl`): Each row represents a patient
  - Required columns: `patient_record_id`, `town`, `hospital`, `iso2`, `<target_column>`
  - Feature columns: Clinical features, demographics, etc.

- **Hospital data** (`df_site_encoded.pkl`): Each row represents a hospital
  - Required columns: `town`, `hospital`, `iso2`, `wb` (World Bank income group)
  - Feature columns: Hospital characteristics, resources, etc.

### Preprocessing Raw Data

If you have raw GECKO data, use the preprocessing script to prepare it:

```bash
python src/preprocess_data.py \
    --input data/raw/gecko.csv \
    --output data/processed
```

This will:
- Extract and encode patient features (demographics, clinical, operative)
- Process hospital/site-level characteristics
- Handle missing values with appropriate imputation
- Create geographic mappings (ISO codes, continents)
- Save processed files to `data/processed/`

**Output files:**
- `df_encoded.pkl`: Patient-level features
- `df_site_encoded.pkl`: Hospital-level features
- `iso_map.pkl`: ISO2 to continent mapping
- `iso_ver_map.pkl`: ISO2 to ISO3 mapping

## Quick Start

### Basic Training (Recommended for First-Time Users)

The simplest way to train the ContextSurg model:

```bash
python src/train.py \
    --data_dir data/processed \
    --target comps_chole_2l \
    --output_dir outputs
```

This will:
- Load data from `data/processed/`
- Train ContextSurg with default settings
- Save model and results to `outputs/`

**Key Arguments:**
- `--target`: Prediction target (`comps_chole_2l`, `comps_major_2l`, `comps_any_2l`)
- `--num_experts`: Number of expert networks (default: 5)
- `--batch_size`: Training batch size (default: 256)
- `--num_epochs`: Number of epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)

### Customizing Hyperparameters

You can customize model hyperparameters via command-line arguments:

```bash
python src/train.py \
    --data_dir data/processed \
    --target comps_chole_2l \
    --num_experts 10 \
    --embedding_dim 128 \
    --batch_size 512 \
    --learning_rate 0.0005 \
    --num_epochs 150
```

See `python src/train.py --help` for all available options.

### 3. Evaluation

The training script automatically evaluates the model on the test set and saves:
- Model performance metrics (AUROC, AUPRC)
- Trained model weights
- Test predictions and labels
- Average feature importance (CSV file)

Results are saved in the `outputs/` directory.

## Configuration

Model hyperparameters are defined directly in `src/train.py` in the `DEFAULT_CONFIG` dictionary:

```python
DEFAULT_CONFIG = {
    # Model architecture
    'num_experts': 5,
    'embedding_dim': 64,
    'hospital_embedding_dim': 128,
    'expert_dim': 128,
    'shared_hidden_dims': [128, 64],
    'router_hidden_dims': [64],
    'expert_hidden_dims': [256, 128],

    # Training parameters
    'batch_size': 256,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,

    # Loss weights
    'lambda_fair': 0.1,
    'lambda_adv': 0.0,
    'lambda_pred': 1.0,
}
```

## Interactive Visualizations

The `country_visualization/` directory contains an interactive web-based visualization tool for exploring country-level model performance.

To view locally:
```bash
cd country_visualization
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

The visualization is also available as a GitHub Pages site (see project website).
<!-- 
## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper2024,
  title={ContextSurg for Healthcare Risk Prediction Across Diverse Settings},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
``` -->
