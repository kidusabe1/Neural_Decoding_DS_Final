# Neural Decoding Pipeline

A comprehensive machine learning pipeline for decoding neural activity from spike trains to predict behavioral outputs (e.g., movement kinematics). This project implements and compares multiple decoding methods ranging from classic linear approaches to modern deep learning techniques, with automated hyperparameter tuning and extensive visualization capabilities.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Key Stages](#key-stages)
- [Data Description](#data-description)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Project](#running-the-project)
- [Notebooks](#notebooks)
- [Running Tests](#running-tests)
- [Output Files](#output-files)
- [References](#references)
- [Contributing](#contributing)

---

## Project Description

### Main Objectives

1. **Implement multiple neural decoding algorithms** to predict continuous behavioral outputs from neural spike data
2. **Compare decoder performance** across different methods and datasets using standardized metrics
3. **Provide automated hyperparameter tuning** using Bayesian optimization for optimal model performance
4. **Deliver a modular, extensible framework** for neural decoding research with comprehensive documentation
5. **Evaluate generalization** across different brain regions (motor cortex M1, hippocampus HC)

### Research Hypothesis

Modern deep learning methods (LSTM, Dense NN) can capture complex nonlinear temporal dynamics in neural encoding that classical methods (Wiener Filter, Kalman Filter) may miss, particularly when sufficient training data is available. However, classical methods may provide better interpretability and computational efficiency.

### Key Assumptions

- Neural activity encodes information about behavioral states
- Spike trains can be binned into discrete time intervals without significant information loss
- Past neural activity contains predictive information about current/future behavioral states
- Both linear and nonlinear relationships exist between neural activity and behavior

---

## Features

✨ **Comprehensive Model Suite**
- 5 decoder implementations: Wiener Filter, Wiener Cascade, Kalman Filter, Dense NN, LSTM
- Automated hyperparameter tuning with Bayesian optimization
- One-shot training for Kalman Filter with optimal parameters

📊 **Rich Visualizations**
- Time series plots comparing true vs predicted outputs
- Performance bar charts across models and datasets
- Scatter plots for correlation analysis
- Automated figure generation and saving

💾 **Persistent Storage**
- Trained models saved as pickle files
- Predictions and metrics saved in multiple formats (NPZ, JSON, CSV)
- Hyperparameters logged for reproducibility

🔧 **Modular Architecture**
- Clean separation of concerns (data, models, evaluation, visualization)
- Easy to extend with new decoders or datasets
- Comprehensive unit tests

📈 **Performance Metrics**
- R² (coefficient of determination)
- Pearson correlation
- RMSE (root mean squared error)
- Per-dimension metrics for multi-output predictions

---

## Folder Structure

```
Neural_Decoding_DS_Final/
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt        # Python package dependencies
├── README.md               # This file - project documentation
├── QUICK_START.md          # Quick reference guide
├── CHANGES_SUMMARY.md      # Detailed change log
├── LICENSE                 # BSD 3-Clause License
│
├── data/
│   ├── raw/                # Raw .mat data files (download required)
│   │   ├── m1_data_raw.mat
│   │   └── hc_data_raw.mat
│   └── processed/          # Preprocessed data (auto-generated)
│
├── src/
│   └── neural_decoding/
│       ├── __init__.py
│       ├── __main__.py     # CLI entry point
│       ├── main.py         # Main pipeline orchestration
│       ├── config.py       # Configuration and hyperparameters
│       ├── logger.py       # Logging setup
│       │
│       ├── data/           # Data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── loader.py   # MAT file loading utilities
│       │   └── preprocessing.py  # Spike binning, train/test splits
│       │
│       ├── models/         # Decoder implementations
│       │   ├── __init__.py
│       │   ├── base.py     # Abstract base decoder class
│       │   ├── wiener.py   # Wiener Filter & Wiener Cascade
│       │   ├── kalman.py   # Kalman Filter (Wu et al. 2003)
│       │   └── neural_nets.py  # Dense NN, LSTM (TensorFlow/Keras)
│       │
│       ├── evaluation/     # Metrics and hyperparameter tuning
│       │   ├── __init__.py
│       │   ├── hyperopt.py # Bayesian optimization utilities
│       │   └── metrics.py  # R², correlation, RMSE
│       │
│       └── visualization/  # Plotting and figures
│           ├── __init__.py
│           └── plots.py    # Time series, scatter, comparison plots
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── model_comparison.ipynb  # Complete training & comparison pipeline
│   ├── Kalman_Filter.ipynb     # Kalman filter deep dive
│   ├── LSTM.ipynb              # LSTM model exploration
│   └── debug.ipynb             # Debugging utilities
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_config.py      # Configuration tests
│   ├── test_main.py        # Pipeline integration tests
│   ├── test_data_preprocessing.py  # Data processing tests
│   ├── test_models.py      # Model implementation tests
│   └── test_evaluation.py  # Metrics and evaluation tests
│
└── reports/                # Generated outputs (auto-created)
    ├── models/             # Saved model files (.pkl) and parameters (.json)
    ├── outputs/            # Predictions (.npz) and metrics (.json, .csv)
    └── figures/            # Generated plots (.png)
```

---

## Key Stages

### Stage 1: Data Import
**Module:** `src/neural_decoding/data/loader.py`

- Load neural spike data and behavioral outputs from pickle/h5 files
- Support for multiple datasets (motor cortex, somatosensory cortex, hippocampus)
- Validate data format and dimensions

### Stage 2: Data Preprocessing
**Module:** `src/neural_decoding/data/preprocessing.py`

- **Spike binning:** Convert spike times to firing rates in time bins
- **Feature extraction:** Create covariate matrix with spike history
- **Train/test split:** Divide data for model training and evaluation

### Stage 3: Model Training
**Module:** `src/neural_decoding/models/`

Train decoders with automated hyperparameter tuning:

#### Available Decoders

| Model | Type | Tuning | Speed | Best For |
|-------|------|--------|-------|----------|
| **Wiener Filter** | Linear regression | None (no hyperparameters) | ⚡⚡⚡ Fast | Baseline comparisons |
| **Wiener Cascade** | Linear-nonlinear | Bayesian Opt (degree) | ⚡⚡ Medium | Nonlinear mappings |
| **Kalman Filter** | State-space | One-shot optimal | ⚡⚡⚡ Fast | Temporal tracking |
| **Dense NN** | Feedforward network | Bayesian Opt (units, dropout, epochs) | ⚡ Slow | Complex patterns |
| **LSTM** | Recurrent network | Bayesian Opt (units, dropout, epochs) | 🐌 Slowest | Temporal sequences |

#### Hyperparameter Tuning
- **Bayesian Optimization:** Efficient search using `bayes_opt` library
- **Validation-based:** Tuning performed on separate validation set
- **Automated saving:** Best parameters saved with trained models

### Stage 4: Evaluation
**Module:** `src/neural_decoding/evaluation/metrics.py`

- **Performance metrics:**
  - R² (coefficient of determination)
  - Pearson correlation coefficient
  - RMSE (root mean squared error)
- **Per-dimension analysis:** Individual metrics for multi-output predictions
- **Cross-model comparison:** Automated comparison across all decoders
- **Statistical validation:** Metrics computed on held-out test set

### Stage 5: Visualization & Reporting
**Module:** `src/neural_decoding/visualization/plots.py`

- **Time series plots:** True vs predicted outputs with smoothing
- **Performance bar charts:** R² comparison across models and datasets
- **Scatter plots:** Predicted vs true values with perfect prediction line
- **Automated saving:** All figures saved to `reports/figures/` in high resolution
- **Summary tables:** CSV exports with all metrics and hyperparameters

---


## Data Description

### Dataset Format

Neural recordings paired with behavioral measurements:

- **Neural data (X):** Spike times or binned firing rates
  - Shape: `(n_time_bins, n_neurons)` after binning
  - Values: Firing rates (spikes/second) or spike counts
  
- **Behavioral data (y):** Continuous output variables
  - Shape: `(n_time_bins, n_output_features)`
  - Examples: velocity (x, y), position, acceleration

### Available Datasets

**Download from:** [Dropbox Link](https://www.dropbox.com/sh/n4924ipcfjqc0t6/AACPWjxDKPEzQiXKUUFriFkJa?dl=0)

| Dataset | Brain Region | Task | Neurons | Outputs | Sampling |
|---------|--------------|------|---------|---------|----------|
| **M1** | Motor Cortex (M1) | Reaching | ~100 | 2D velocity | 50ms bins |
| **HC** | Hippocampus | Spatial navigation | ~50 | 2D position | 200ms bins |

### Data Format

Files are MATLAB `.mat` format containing:
- `spike_times`: List of spike time arrays (one per neuron)
- `vels` or `pos`: Behavioral outputs with timestamps
- Automatically loaded via `neural_decoding.data.load_dataset()`

---

## Installation

### Prerequisites
- **Python:** 3.9 or higher
- **Package manager:** pip or conda
- **Optional:** CUDA-capable GPU for neural network training

### Quick Install

1. **Clone the repository:**
```bash
git clone https://github.com/kidusabe1/Neural_Decoding_DS_Final.git
cd Neural_Decoding_DS_Final
```

2. **Create virtual environment:**
```bash
python -m venv .venv

# Activate (macOS/Linux):
source .venv/bin/activate

# Activate (Windows):
.venv\Scripts\activate
```

3. **Install package:**
```bash
# Full installation with all features
pip install -e ".[dev,notebook,optimization]"

# Or minimal installation (core features only)
pip install -e .

# Or from requirements.txt
pip install -r requirements.txt
```

### Install Options

| Command | Includes |
|---------|----------|
| `pip install -e .` | Core package (Wiener, Kalman only) |
| `pip install -e ".[dev]"` | Core + testing tools |
| `pip install -e ".[notebook]"` | Core + Jupyter support |
| `pip install -e ".[optimization]"` | Core + Bayesian optimization |
| `pip install -e ".[dev,notebook,optimization]"` | Everything |

### Download Data

```bash
# Create data directory
mkdir -p data/raw

# Download datasets from Dropbox (link in Data Description)
# Place .mat files in data/raw/
```

---

## Quick Start

### Option 1: Use Notebooks (Recommended)

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/model_comparison.ipynb
# Run all cells to train all models and generate visualizations
```

### Option 2: Command Line Interface

```bash
# Train single model
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder kalman

# Train with custom parameters
python -m neural_decoding \
  --data_path data/raw/hc_data_raw.mat \
  --decoder lstm \
  --bin_size 0.2 \
  --test_size 0.2 \
  --verbose
```

### Option 3: Python API

```python
from pathlib import Path
from neural_decoding.data import load_dataset, bin_spikes, bin_output, prepare_train_test_split
from neural_decoding.models import KalmanFilterDecoder
from neural_decoding.evaluation.metrics import evaluate_decoder

# Load data
data_path = Path("data/raw/m1_data_raw.mat")
raw = load_dataset(data_path)

# Bin spikes
neural = bin_spikes(raw["spike_times"], bin_size=0.05, end_time=300)
outputs = bin_output(raw["outputs"], raw["output_times"], bin_size=0.05, end_time=300)

# Prepare train/test split
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_train_test_split(
    neural, outputs, bins_before=13, bins_after=0, bins_current=1, test_size=0.2
)

# Train model
model = KalmanFilterDecoder(noise_scale_c=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = evaluate_decoder(y_test, y_pred)
print(f"R² = {metrics['r2']:.4f}")
```

---

## Notebooks

Interactive Jupyter notebooks for exploration and analysis:

### 📊 `model_comparison.ipynb` (Main Notebook)
**Complete training and comparison pipeline**
- Trains all 5 decoder types (Wiener Filter, Wiener Cascade, Kalman, Dense NN, LSTM)
- Automated hyperparameter tuning with Bayesian optimization
- Saves trained models, predictions, and metrics
- Generates comprehensive visualizations
- **Outputs:** Saved to `reports/models/`, `reports/outputs/`, `reports/figures/`

### 🎯 `Kalman_Filter.ipynb`
Deep dive into Kalman Filter implementation
- Detailed explanation of state-space modeling
- Step-by-step algorithm walkthrough
- Sensitivity analysis

### 🧠 `LSTM.ipynb`
LSTM model exploration and analysis
- Architecture design choices
- Training dynamics visualization
- Sequence modeling insights

### 🔧 `debug.ipynb`
Debugging and development utilities
- Data inspection tools
- Model diagnostics
- Quick testing

---

## Running Tests

Comprehensive test suite for reliability:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage report
pytest --cov=neural_decoding --cov-report=html

# Run tests for specific module
pytest tests/test_data_preprocessing.py::test_bin_spikes -v
```

### Test Coverage

- `test_config.py` - Configuration and path management
- `test_data_preprocessing.py` - Data loading and binning
- `test_models.py` - Decoder implementations
- `test_evaluation.py` - Metrics and evaluation
- `test_main.py` - End-to-end pipeline integration

---

## Output Files

All outputs are automatically organized in the `reports/` directory:

### Saved Models (`reports/models/`)
```
wiener_filter_m1.pkl          # Trained model (pickle)
wiener_filter_m1_params.json  # Hyperparameters
kalman_filter_hc.pkl
kalman_filter_hc_params.json
lstm_m1.pkl
lstm_m1_params.json
...
```

### Predictions (`reports/outputs/`)
```
wiener_filter_m1_predictions.npz  # Test predictions (NumPy)
wiener_filter_m1_metrics.json     # Performance metrics
model_comparison_summary.csv      # Summary table
...
```

### Figures (`reports/figures/`)
```
model_comparison_grid.png         # Main comparison plot
performance_comparison_bars.png   # Bar chart
scatter_plots.png                 # Scatter analysis
...
```

### Loading Saved Results

```python
import pickle
import numpy as np
import json

# Load trained model
with open('reports/models/kalman_filter_m1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load predictions
data = np.load('reports/outputs/kalman_filter_m1_predictions.npz')
y_test = data['y_test']
y_pred = data['y_pred']

# Load metrics
with open('reports/outputs/kalman_filter_m1_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"R² = {metrics['metrics']['r2']}")
```

---

## References

### Key Papers

1. **Glaser, J. I., Benjamin, A. S., Farhoodi, R., & Kording, K. P. (2019).** The roles of supervised machine learning in systems neuroscience. *Progress in Neurobiology*, 175, 126-137. [arXiv:1708.00909](https://arxiv.org/abs/1708.00909)

2. **Wu, W., et al. (2003).** Neural decoding of cursor motion using a Kalman filter. *Advances in Neural Information Processing Systems*, 15.

3. **Zhang, K., et al. (1998).** Interpreting neuronal population activity by reconstruction. *Journal of Neurophysiology*, 79(2), 1017-1044.

### Code References

- **Original Implementation:** [KordingLab/Neural_Decoding](https://github.com/KordingLab/Neural_Decoding)
- **Bayesian Optimization:** [fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

### Related Resources

- [Neural Decoding Review (Glaser et al.)](https://arxiv.org/abs/1708.00909)
- [Kalman Filter Tutorial](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [LSTM for Time Series](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Add tests for new features
   - Update documentation
   - Follow PEP 8 style guide
4. **Run tests**
   ```bash
   pytest
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Write unit tests for all new features
- Maintain backwards compatibility
- Document all public APIs
- Use type hints where appropriate
- Keep commits atomic and well-described

---

## License

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_decoding_2026,
  title={Neural Decoding Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/kidusabe1/Neural_Decoding_DS_Final}
}
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/kidusabe1/Neural_Decoding_DS_Final/issues)
- **Discussions:** [GitHub Discussions](https://github.com/kidusabe1/Neural_Decoding_DS_Final/discussions)
- **Documentation:** See [QUICK_START.md](QUICK_START.md) for quick reference

---

## Acknowledgments

- Original Neural_Decoding package by [KordingLab](https://github.com/KordingLab)
- Dataset providers and neuroscience community
- Open-source contributors

---

**Last Updated:** January 2026
