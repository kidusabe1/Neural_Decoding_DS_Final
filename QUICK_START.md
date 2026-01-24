# Quick Start Guide - Neural Decoding

This guide provides quick commands and examples to get you started with the Neural Decoding pipeline.

---

## 🚀 Installation (30 seconds)

```bash
# Clone and enter directory
git clone https://github.com/kidusabe1/Neural_Decoding_DS_Final.git
cd Neural_Decoding_DS_Final

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e ".[dev,notebook,optimization]"

# Download data to data/raw/ (see README for links)
```

---

## 📊 Quick Usage

### Option 1: Use Jupyter Notebooks (Recommended)

```bash
# Launch Jupyter
jupyter notebook

# Open and run notebooks/model_comparison.ipynb
# This trains all models, saves results, and generates plots
```

**What it does:**
- ✅ Trains 5 decoder types (Wiener Filter, Wiener Cascade, Kalman, Dense NN, LSTM)
- ✅ Auto-tunes hyperparameters (Bayesian optimization)
- ✅ Saves models to `reports/models/`
- ✅ Saves predictions to `reports/outputs/`
- ✅ Generates plots in `reports/figures/`

### Option 2: Command Line Interface

```bash
# Basic usage - train Kalman filter on M1 dataset
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder kalman

# Train LSTM with custom settings
python -m neural_decoding \
  --data_path data/raw/hc_data_raw.mat \
  --decoder lstm \
  --bin_size 0.2 \
  --test_size 0.2 \
  --verbose

# See all options
python -m neural_decoding --help
```

### Option 3: Python API

```python
from pathlib import Path
from neural_decoding.data import load_dataset, bin_spikes, bin_output, prepare_train_test_split
from neural_decoding.models import KalmanFilterDecoder, LSTMDecoder
from neural_decoding.evaluation.metrics import evaluate_decoder

# Load data
raw = load_dataset(Path("data/raw/m1_data_raw.mat"))

# Bin neural data
neural = bin_spikes(raw["spike_times"], bin_size=0.05, end_time=300)
outputs = bin_output(raw["outputs"], raw["output_times"], bin_size=0.05, end_time=300)

# Split data
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_train_test_split(
    neural, outputs, bins_before=13, bins_after=0, test_size=0.2
)

# Train Kalman Filter (fast, optimal parameters)
kalman = KalmanFilterDecoder(noise_scale_c=1.0)
kalman.fit(X_train, y_train)
y_pred = kalman.predict(X_test)
metrics = evaluate_decoder(y_test, y_pred)
print(f"Kalman R² = {metrics['r2']:.4f}")

# Train LSTM (slower, more accurate)
lstm = LSTMDecoder(units=128, dropout_rate=0.25, num_epochs=20, verbose=1)
lstm.fit(X_train, y_train)
y_pred_lstm = lstm.predict(X_test)
metrics_lstm = evaluate_decoder(y_test, y_pred_lstm)
print(f"LSTM R² = {metrics_lstm['r2']:.4f}")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output and coverage
pytest -v --cov=neural_decoding --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_data_preprocessing.py::test_bin_spikes -v

# View coverage report
open htmlcov/index.html  # macOS
# or: xdg-open htmlcov/index.html  # Linux
# or: start htmlcov/index.html  # Windows
```

---

## 🎯 Available Decoders

| Decoder | Speed | Tuning | Best For | Command |
|---------|-------|--------|----------|---------|
| **Wiener Filter** | ⚡⚡⚡ | None | Baseline, quick tests | `--decoder wiener_filter` |
| **Wiener Cascade** | ⚡⚡ | Bayesian Opt | Nonlinear mappings | `--decoder wiener_cascade` |
| **Kalman Filter** | ⚡⚡⚡ | One-shot optimal | Real-time tracking | `--decoder kalman` |
| **Dense NN** | ⚡ | Bayesian Opt | Complex patterns | `--decoder dense_nn` |
| **LSTM** | 🐌 | Bayesian Opt | Temporal sequences | `--decoder lstm` |

---

## 📁 Output Files

---

## 📁 Output Files

After running the pipeline, outputs are organized in `reports/`:

```
reports/
├── models/                              # Trained models
│   ├── kalman_filter_m1.pkl            # Serialized model
│   ├── kalman_filter_m1_params.json    # Hyperparameters
│   ├── lstm_hc.pkl
│   └── ...
│
├── outputs/                             # Predictions and metrics
│   ├── kalman_filter_m1_predictions.npz  # Test set predictions
│   ├── kalman_filter_m1_metrics.json     # Performance metrics
│   ├── model_comparison_summary.csv      # Summary table
│   └── ...
│
└── figures/                             # Generated plots
    ├── model_comparison_grid.png        # Main comparison plot
    ├── performance_comparison_bars.png  # Bar chart
    ├── scatter_plots.png                # Scatter analysis
    └── ...
```

### Loading Results

```python
import pickle
import numpy as np
import json
import pandas as pd

# Load trained model
with open('reports/models/kalman_filter_m1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load predictions
data = np.load('reports/outputs/kalman_filter_m1_predictions.npz')
y_test = data['y_test']
y_pred = data['y_pred']

# Load metrics
with open('reports/outputs/kalman_filter_m1_metrics.json') as f:
    metrics = json.load(f)
    print(f"R² = {metrics['metrics']['r2']}")

# Load summary table
df = pd.read_csv('reports/outputs/model_comparison_summary.csv')
print(df)
```

---

## ⚙️ Configuration

### Command Line Arguments

```bash
python -m neural_decoding --help

Arguments:
  --data_path PATH      Path to .mat data file (required)
  --decoder TYPE        Decoder: wiener_filter, wiener_cascade, kalman, dense_nn, lstm
  --bin_size FLOAT      Spike bin duration in seconds (default: 0.05)
  --test_size FLOAT     Test set proportion (default: 0.2)
  --validation_size     Validation set proportion (default: 0.1)
  --output_dir PATH     Output directory (default: reports/)
  --verbose            Enable verbose logging
  --seed INT           Random seed (default: 42)
```

### Python Configuration

```python
from neural_decoding.config import Paths, DecodingConfig

# Get project paths
paths = Paths.from_here()
print(paths.data_raw)       # .../data/raw
print(paths.reports)        # .../reports

# Modify config
config = DecodingConfig()
config.data.bin_size = 0.1
config.data.test_size = 0.25
config.training.random_seed = 123
```

### Dataset-Specific Settings

The `model_comparison.ipynb` notebook uses optimized settings per dataset:

```python
binning_config = {
    "M1": {
        "bin_size": 0.05,      # 50ms bins
        "bins_before": 13,      # Use 13 past bins
        "bins_current": 1,
        "bins_after": 0
    },
    "HC": {
        "bin_size": 0.20,      # 200ms bins
        "bins_before": 4,       # Use 4 past bins
        "bins_current": 1,
        "bins_after": 5         # Use 5 future bins
    }
}
```

---

## 🔍 Common Tasks

### 1. Quick Test with Wiener Filter (Fastest)
```bash
python -m neural_decoding \
  --data_path data/raw/m1_data_raw.mat \
  --decoder wiener_filter
```

### 2. Train Kalman Filter (Good Balance)
```bash
python -m neural_decoding \
  --data_path data/raw/hc_data_raw.mat \
  --decoder kalman \
  --bin_size 0.2
```

### 3. Train LSTM for Best Accuracy
```bash
python -m neural_decoding \
  --data_path data/raw/m1_data_raw.mat \
  --decoder lstm \
  --verbose
```

### 4. Compare All Models (Use Notebook)
```bash
jupyter notebook notebooks/model_comparison.ipynb
# Run all cells
```

### 5. Load and Use Saved Model
```python
import pickle
import numpy as np

# Load model
with open('reports/models/kalman_filter_m1.pkl', 'rb') as f:
    model = pickle.load(f)

# Use for prediction
X_new = np.random.randn(100, 150)  # Example input
y_pred = model.predict(X_new)
```

### 6. Visualize Results
```python
from neural_decoding.visualization.plots import plot_predictions
import matplotlib.pyplot as plt
import numpy as np

# Load predictions
data = np.load('reports/outputs/kalman_filter_m1_predictions.npz')
y_test = data['y_test']
y_pred = data['y_pred']

# Plot
fig = plot_predictions(
    y_true=y_test,
    y_pred=y_pred,
    output_names=['X velocity', 'Y velocity'],
    title='Kalman Filter Predictions'
)
plt.show()
```

---

## 🐛 Troubleshooting

### TensorFlow/Neural Network Issues

**Problem:** ImportError for LSTM or Dense NN
```bash
# Solution: Install TensorFlow
pip install tensorflow

# Or install with GPU support
pip install tensorflow-gpu  # Requires CUDA
```

**Problem:** TensorFlow version incompatibility
```bash
# Use specific version
pip install tensorflow==2.13.0
```

### Data Issues

**Problem:** File not found error
```bash
# Check file exists
ls data/raw/*.mat

# Verify path is correct
python -c "from pathlib import Path; print(Path('data/raw/m1_data_raw.mat').exists())"
```

**Problem:** Data format error
```python
# Inspect data structure
from neural_decoding.data import load_dataset
raw = load_dataset(Path("data/raw/m1_data_raw.mat"))
print(raw.keys())  # Should show: spike_times, vels, vel_times (or similar)
```

### Memory Issues

**Problem:** Out of memory during LSTM training
```python
# Reduce batch size
lstm = LSTMDecoder(units=64, batch_size=64, num_epochs=20)

# Or reduce data size
X_train = X_train[:10000]  # Use subset
y_train = y_train[:10000]
```

**Problem:** Notebook kernel dies
```bash
# Increase memory limit or restart kernel
# Run cells sequentially instead of all at once
```

### Performance Issues

**Problem:** Training too slow
```bash
# Use faster decoder
python -m neural_decoding --decoder wiener_filter  # Fastest

# Or increase bin size
python -m neural_decoding --bin_size 0.1  # Fewer time bins
```

**Problem:** Poor prediction accuracy
```bash
# Try different decoder
python -m neural_decoding --decoder lstm  # Most accurate

# Or tune hyperparameters (use notebook for auto-tuning)
jupyter notebook notebooks/model_comparison.ipynb
```

### Package/Import Issues

**Problem:** Module not found
```bash
# Reinstall package in development mode
pip install -e .

# Verify installation
python -c "import neural_decoding; print(neural_decoding.__version__)"
```

**Problem:** Permission errors
```bash
# Check directory permissions
ls -la

# Create missing directories
mkdir -p data/raw reports/models reports/outputs reports/figures
```

---

## 💡 Tips & Best Practices

### Performance Optimization

1. **Start with Wiener Filter** for quick baseline (< 1 second)
2. **Use Kalman Filter** for good speed/accuracy tradeoff
3. **Reserve LSTM** for final tuned model (requires patience)
4. **Increase bin_size** to reduce computational load
5. **Use GPU** for neural network training when available

### Reproducibility

```python
# Always set random seeds
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# For TensorFlow models
import tensorflow as tf
tf.random.set_seed(42)
```

### Model Selection Guide

| Use Case | Recommended Decoder | Why |
|----------|-------------------|-----|
| Real-time decoding | Kalman Filter | Fast inference, online updates |
| Offline analysis | LSTM | Best accuracy, captures dynamics |
| Quick prototyping | Wiener Filter | Instant results, simple |
| Nonlinear mapping | Wiener Cascade | Captures static nonlinearities |
| Multiple comparisons | Run notebook | Trains all models, auto-tunes |

### Data Preparation Tips

```python
# Check data quality before training
from neural_decoding.data import load_dataset
import numpy as np

raw = load_dataset("data/raw/m1_data_raw.mat")

# Inspect spike times
print(f"Number of neurons: {len(raw['spike_times'])}")
for i, spikes in enumerate(raw['spike_times'][:5]):
    print(f"Neuron {i}: {len(spikes)} spikes")

# Check for missing data
outputs = raw['outputs']
print(f"Output shape: {outputs.shape}")
print(f"NaN values: {np.isnan(outputs).sum()}")
```

### Hyperparameter Tuning

The `model_comparison.ipynb` notebook automatically tunes:
- **Wiener Cascade:** polynomial degree (2-5)
- **Dense NN:** units (64-256), dropout (0.05-0.5), epochs (15-40)
- **LSTM:** units (32-256), dropout (0.05-0.5), epochs (15-40)
- **Kalman Filter:** Uses optimal parameters (no tuning needed)

---

## 📚 Additional Resources

### Project Documentation
- **Full Documentation:** [README.md](README.md)
- **Change Log:** [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- **API Reference:** See docstrings in source code

### Notebooks
- `model_comparison.ipynb` - Complete training pipeline ⭐
- `Kalman_Filter.ipynb` - Kalman filter deep dive
- `LSTM.ipynb` - LSTM model exploration
- `debug.ipynb` - Debugging utilities

### External Links
- [KordingLab Neural Decoding](https://github.com/KordingLab/Neural_Decoding) - Original implementation
- [Glaser et al. (2019)](https://arxiv.org/abs/1708.00909) - Neural decoding review
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization) - Hyperparameter tuning library

### Getting Help
- **GitHub Issues:** [Report bugs or request features](https://github.com/kidusabe1/Neural_Decoding_DS_Final/issues)
- **Discussions:** [Ask questions](https://github.com/kidusabe1/Neural_Decoding_DS_Final/discussions)

---

## 🎓 Example Workflows

### Research Workflow

```bash
# 1. Install and setup
pip install -e ".[dev,notebook,optimization]"

# 2. Run comprehensive analysis
jupyter notebook notebooks/model_comparison.ipynb

# 3. Analyze results
python
>>> import pandas as pd
>>> df = pd.read_csv('reports/outputs/model_comparison_summary.csv')
>>> print(df.groupby('Dataset')['R²'].describe())

# 4. Generate paper figures
# All figures saved in reports/figures/
```

### Production Deployment

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train optimized model
jupyter nbconvert --execute notebooks/model_comparison.ipynb

# 3. Export best model
python
>>> import pickle
>>> with open('reports/models/lstm_m1.pkl', 'rb') as f:
>>>     model = pickle.load(f)

# 4. Use in production
>>> y_pred = model.predict(X_new)
```

---

## 🔑 Key Commands Cheatsheet

```bash
# Installation
pip install -e ".[dev,notebook,optimization]"

# Run notebook
jupyter notebook notebooks/model_comparison.ipynb

# CLI training
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder kalman

# Testing
pytest -v --cov=neural_decoding

# View results
ls reports/figures/
ls reports/models/
ls reports/outputs/
```

---

For detailed information, see the [full README](README.md).

**Last Updated:** January 2026
