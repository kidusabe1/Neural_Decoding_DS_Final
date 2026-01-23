# Quick Reference Guide - Neural_Decoding CLI

## Running the Application

### Basic Usage
```bash
# Simple command with default parameters
python -m neural_decoding --data_path data/raw/m1_data_raw.mat
```

### Advanced Usage Examples
```bash
# Kalman filter with custom settings
python -m neural_decoding \
  --data_path data/raw/hc_data_raw.mat \
  --decoder kalman \
  --bin_size 0.1 \
  --test_size 0.3 \
  --output_dir reports/kalman_results

# Wiener cascade decoder
python -m neural_decoding \
  --data_path data/raw/m1_data_raw.mat \
  --decoder wiener_cascade \
  --bin_size 0.05

# With verbose output
python -m neural_decoding \
  --data_path data/raw/m1_data_raw.mat \
  --verbose
```

### Available Decoders
- `wiener_filter` (default) - Fast linear decoder
- `wiener_cascade` - Nonlinear Wiener filter
- `kalman` - Optimal tracking filter
- `dense_nn` - Neural network (requires TensorFlow)
- `lstm` - LSTM recurrent network (requires TensorFlow)

### Help Command
```bash
python -m neural_decoding --help
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py -v

# Run tests with coverage
pytest tests/ --cov=neural_decoding

# Run a specific test
pytest tests/test_config.py::test_paths_from_here -v
```

---

## Logging

All application logs are automatically saved to `app.log`:
```bash
# View logs in real-time
tail -f app.log

# Search for errors
grep ERROR app.log

# View last 50 lines
tail -50 app.log
```

---

## Configuration

Configuration is managed through dataclasses in `src/neural_decoding/config.py`:

```python
from neural_decoding.config import Paths, DecodingConfig

# Get project paths
paths = Paths.from_here()
print(paths.data_raw)          # project/data/raw
print(paths.data_processed)    # project/data/processed

# Access or modify default config
config = DecodingConfig()
config.data.bin_size = 0.1
config.data.test_size = 0.25
```

---

## Project Structure

```
Neural_Decoding_DS_Final/
├── src/neural_decoding/        # Main package
│   ├── __init__.py             # Package entry point
│   ├── __main__.py             # CLI entry point
│   ├── main.py                 # Main pipeline
│   ├── config.py               # Configuration & paths
│   ├── logger.py               # Logging setup
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Decoder implementations
│   ├── evaluation/             # Metrics & evaluation
│   └── visualization/          # Plotting utilities
├── tests/                      # Unit tests
│   ├── test_config.py
│   ├── test_main.py
│   └── __init__.py
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed datasets
├── reports/                    # Output figures & results
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Dependencies
├── CHANGES_SUMMARY.md          # Detailed change log
└── app.log                     # Application logs (auto-generated)
```

---

## Common Tasks

### Check if data loads correctly
```bash
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder wiener_filter
```

### Compare different decoders
```bash
# Run Wiener filter
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder wiener_filter

# Run Kalman filter
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder kalman

# View results in reports/figures/
```

### Modify pipeline parameters
Edit pipeline settings by passing command-line arguments. All available options:
- `--data_path` - Path to .mat file (required)
- `--decoder` - Decoder type
- `--bin_size` - Spike bin duration (seconds)
- `--test_size` - Train/test split ratio
- `--output_dir` - Output directory for figures
- `--bayes_opt` - Enable Bayesian hyperparameter optimization
- `--verbose` - Verbose logging

---

## Troubleshooting

### ImportError for TensorFlow models
```bash
# Install TensorFlow first
pip install tensorflow

# Then run LSTM decoder
python -m neural_decoding --data_path data/raw/m1_data_raw.mat --decoder lstm
```

### Data file not found
- Ensure path is correct: `python -m neural_decoding --help` shows examples
- Check file exists: `ls data/raw/*.mat`

### Permission errors on app.log
- Check current directory permissions: `ls -la app.log`
- Ensure write access to project directory

---

## Performance Tips

- **Use Wiener filter** for quick testing (fastest)
- **Use Kalman filter** for real-time applications (good speed/accuracy)
- **Use neural networks** for best accuracy (slowest, requires GPU)
- **Increase bin_size** for lower temporal resolution (faster processing)
- **Decrease test_size** for more training data (better accuracy)

---

## Production Deployment

For deploying this application:

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Run Pipeline**
   ```bash
   python -m neural_decoding --data_path /path/to/data.mat
   ```

3. **Monitor Logs**
   ```bash
   tail -f app.log
   ```

4. **Collect Results**
   - Figures: `reports/figures/*.png`
   - Logs: `app.log`

---

For detailed information about changes made, see [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
