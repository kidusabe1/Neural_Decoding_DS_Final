# Neural Decoding Pipeline

A comprehensive machine learning pipeline for decoding neural activity from spike trains to predict behavioral outputs (e.g., movement kinematics). This project implements and compares multiple decoding methods ranging from classic linear approaches to modern deep learning techniques.

---

## Table of Contents

- [Project Description](#project-description)
- [Objectives and Hypothesis](#objectives-and-hypothesis)
- [Folder Structure](#folder-structure)
- [Key Stages](#key-stages)
- [Key Parameters and Configurations](#key-parameters-and-configurations)
- [Data Description](#data-description)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Running Tests](#running-tests)
- [References](#references)

---

## Project Description

### Main Objectives

1. **Implement multiple neural decoding algorithms** to predict continuous behavioral outputs from neural spike data
2. **Compare decoder performance** across different methods and datasets
3. **Provide a modular, extensible framework** for neural decoding research
4. **Evaluate generalization** across different brain regions (motor cortex, somatosensory cortex, hippocampus)

### Assumptions

- Neural activity encodes information about behavioral states
- Spike trains can be binned into discrete time intervals without significant information loss
- Past neural activity contains predictive information about current/future behavioral states
- Linear and nonlinear relationships exist between neural activity and behavior

### Hypothesis

Deep learning methods (LSTM, GRU) will outperform classical methods (Wiener Filter, Kalman Filter) when sufficient training data is available, particularly for capturing nonlinear dynamics in neural encoding.

---

## Folder Structure

```
Neural_Decoding_DS_Final/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project documentation
├── app.log                 # Application log file (generated at runtime)
│
├── data/
│   ├── raw/                # Raw data files (downloaded datasets)
│   └── processed/          # Preprocessed data ready for modeling
│
├── src/
│   └── neural_decoding/
│       ├── __init__.py
│       ├── config.py       # Configuration and hyperparameters
│       ├── logger.py       # Logging setup
│       │
│       ├── data/           # Data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── loader.py   # Data loading utilities
│       │   └── preprocessing.py  # Spike binning, feature extraction
│       │
│       ├── models/         # Decoder implementations
│       │   ├── __init__.py
│       │   ├── base.py     # Base decoder class
│       │   ├── wiener.py   # Wiener Filter & Cascade
│       │   ├── kalman.py   # Kalman Filter
│       │   ├── svr.py      # Support Vector Regression
│       │   ├── xgboost_decoder.py  # XGBoost
│       │   └── neural_nets.py      # DNN, RNN, GRU, LSTM
│       │
│       ├── evaluation/     # Metrics and analysis
│       │   ├── __init__.py
│       │   └── metrics.py  # R², correlation, etc.
│       │
│       └── visualization/  # Plotting and figures
│           ├── __init__.py
│           └── plots.py    # Visualization utilities
│
├── notebooks/              # Jupyter notebooks for exploration
│   └── example_analysis.ipynb
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
└── reports/                # Generated figures and results
    └── figures/
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
- **Normalization:** Z-score outputs for certain decoders (SVR)

### Stage 3: Model Training
**Module:** `src/neural_decoding/models/`

- Train selected decoder(s) on preprocessed neural data
- Support for hyperparameter optimization
- Available decoders:
  - **Wiener Filter** - Linear regression baseline
  - **Wiener Cascade** - Linear-nonlinear model
  - **Kalman Filter** - State-space model
  - **SVR** - Support Vector Regression
  - **XGBoost** - Gradient boosted trees
  - **Dense NN** - Feedforward neural network
  - **RNN/GRU/LSTM** - Recurrent neural networks

### Stage 4: Evaluation
**Module:** `src/neural_decoding/evaluation/metrics.py`

- Compute performance metrics (R², Pearson correlation)
- Cross-validation across multiple folds
- Compare decoder performance

### Stage 5: Visualization
**Module:** `src/neural_decoding/visualization/plots.py`

- Plot predicted vs. actual behavioral outputs
- Generate performance comparison figures
- Save figures to `reports/figures/`

---


## Data Description

### Dataset Format

- **Neural data:** Matrix of shape `(n_time_bins, n_neurons)` - firing rates
- **Output data:** Matrix of shape `(n_time_bins, n_output_features)` - behavioral variables (e.g., velocity)

### Available Datasets

Data can be downloaded from: [Dropbox Link](https://www.dropbox.com/sh/n4924ipcfjqc0t6/AACPWjxDKPEzQiXKUUFriFkJa?dl=0)

| Dataset | Brain Region | Description |
|---------|--------------|-------------|
| S1 | Somatosensory Cortex | Reaching task kinematics |
| M1 | Motor Cortex | Reaching task kinematics |
| HC | Hippocampus | Spatial navigation |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kidusabe1/Neural_Decoding_DS_Final.git
cd Neural_Decoding_DS_Final
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

3. Install the package:
```bash
# Install with all dependencies
pip install -e ".[dev,notebook,optimization]"

# Or install only core dependencies
pip install -e .
```

---

## Running the Project

### Basic Usage

Run the complete pipeline:
```bash
python -m neural_decoding.main --dataset s1 --decoder lstm
```

### Command Line Options

```bash
python -m neural_decoding.main --help

Options:
  --dataset     Dataset to use: s1, m1, or hc (default: s1)
  --decoder     Decoder to run: wiener, kalman, svr, xgboost, lstm, all (default: all)
  --data-path   Path to data directory (default: data/raw/)
  --output-dir  Directory for results (default: reports/)
  --seed        Random seed for reproducibility (default: 42)
```

### Example Workflow

```python
from neural_decoding.data import load_dataset, preprocess_data
from neural_decoding.models import LSTMDecoder
from neural_decoding.evaluation import get_r2_score

# Load data
neural_data, outputs = load_dataset("data/raw/example_data_s1.pickle")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(
    neural_data, outputs,
    bins_before=13, bins_current=1, bins_after=0
)

# Train decoder
decoder = LSTMDecoder(units=400, num_epochs=5)
decoder.fit(X_train, y_train)

# Evaluate
predictions = decoder.predict(X_test)
r2 = get_r2_score(y_test, predictions)
print(f"R² Score: {r2:.4f}")
```

---

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src/neural_decoding --cov-report=html
```

Run specific test modules:
```bash
pytest tests/test_models.py -v
pytest tests/test_preprocessing.py -v
```

---

## References

### Papers

1. Glaser, J. I., Benjamin, A. S., Farhoodi, R., & Kording, K. P. (2019). **The roles of supervised machine learning in systems neuroscience.** *Progress in Neurobiology*, 175, 126-137. [arXiv:1708.00909](https://arxiv.org/abs/1708.00909)

2. Wu, W., et al. (2003). **Neural decoding of cursor motion using a Kalman filter.** *Advances in Neural Information Processing Systems*, 15.

3. Zhang, K., et al. (1998). **Interpreting neuronal population activity by reconstruction.** *Journal of Neurophysiology*, 79(2), 1017-1044.

### Code References

- Original Neural Decoding Package: [KordingLab/Neural_Decoding](https://github.com/KordingLab/Neural_Decoding)

---

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
