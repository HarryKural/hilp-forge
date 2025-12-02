# HILP-Forge

## Data Augmentation for HILP Event Characterization Using Diffusion Models

Data augmentation for High-Impact Low-Probability (HILP) storm events using deep learning diffusion models. This project synthesizes realistic tornado events to support resilience analysis and emergency planning.

## Project Overview

- **Objective**: Generate synthetic HILP tornado events using state-of-the-art diffusion models
- **Dataset**: NOAA Storm Events Database (2010-2020)
- **Method**: Denoising Diffusion Probabilistic Models (DDPM) for tabular data
- **Applications**: Risk assessment, emergency planning, resilience analysis

## Features

- Automated data collection from NOAA database
- Comprehensive data preprocessing pipeline
- Feature engineering
- HILP event identification (top 5% by impact)
- Diffusion model implementation for data augmentation
- Extensive evaluation metrics and visualizations

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 8GB RAM minimum
- CUDA-capable GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HarryKural/hilp-forge
cd hilp-forge
```

2. **Create virtual environment**
```bash
python3 -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download and preprocess data**
```bash
# Download NOAA data (15-20 minutes)
python run_pipeline.py

# This creates:
# - data/raw/ (downloaded CSV files)
# - data/processed/ (cleaned datasets)
```

### Training the Model
```bash
# Run complete augmentation pipeline
python run_augmentation_pipeline.py

# Or step-by-step:
cd src/augmentation
python tornado_data_prep.py      # prepare data
python diffusion_model.py        # train model
python evaluate_synthetic.py     # evaluate results
```

## Key Components

### Data Preprocessing
- Collects 650K+ storm events from NOAA
- Identifies 32K+ HILP events (top 5% by impact)
- Engineers features including temporal, spatial, and impact metrics

### Diffusion Model
- MLP-based architecture with sinusoidal time embeddings
- 1000 diffusion timesteps
- Trained on normalized tabular data
- Generates realistic synthetic samples via iterative denoising

### Evaluation
- Statistical tests (KS, Wasserstein, Jensen-Shannon)
- Distribution comparisons
- Correlation preservation analysis
- Q-Q plots and visual comparisons

## Usage Examples

### Generate Synthetic Tornadoes
```python
from diffusion_model import load_trained_model
import pandas as pd

# load model
model, trainer = load_trained_model('models/tornado_diffusion.pt')

# generate 1000 synthetic samples
samples = trainer.sample(num_samples=1000)

# save
synthetic_df = pd.DataFrame(samples, columns=feature_names)
synthetic_df.to_csv('synthetic_tornadoes.csv', index=False)
```

### Evaluate
```python
from evaluate_synthetic import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator(
    real_data=real_data,
    synthetic_data=synthetic_data,
    feature_names=feature_names,
    scaler=scaler
)

metrics = evaluator.generate_evaluation_report()
```

## Methodology

1. **Data Collection**: Download NOAA Storm Events (2010-2020)
2. **Preprocessing**: Clean, merge, and engineer features
3. **HILP Identification**: Select top 5% events by composite impact score
4. **Feature Selection**: 9 key features (duration, damage, casualties, etc.)
5. **Normalization**: StandardScaler for zero mean, unit variance
6. **Model Training**: DDPM with 1000 epochs, batch size 64
7. **Generation**: Sample from learned distribution
8. **Evaluation**: Statistical tests and visualizations
