# Examples

This directory contains example scripts demonstrating how to use the Muscle Fatigue Tracker with real and synthetic data.

## Available Examples

### 1. load_zenodo_data.py

**Purpose:** Complete workflow for loading and training models on the Zenodo EMG dataset (Record 14182446).

**What it does:**
- Downloads the Zenodo dataset (or provides manual download instructions)
- Loads EMG signals and fatigue labels
- Preprocesses the data (filtering, normalization)
- Extracts features from EMG signals
- Trains multiple ML models (KNN, SVM, Logistic Regression)
- Evaluates and compares model performance
- Saves the best model for future use

**Usage:**
```bash
python examples/load_zenodo_data.py
```

**Requirements:**
- The Zenodo dataset files should be in `data/zenodo/` directory
- If not present, the script will provide download instructions

**Output:**
- Trained model saved to `models/zenodo_fatigue_model.pkl`
- Performance metrics printed to console
- Model comparison (optional)

**What you'll learn:**
- How to load real EMG data from Zenodo
- Complete preprocessing and feature extraction workflow
- Training and evaluating multiple models
- Model comparison and selection

## Running the Examples

### Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Example Workflow

1. **Download the Zenodo Dataset:**
   - Visit: https://zenodo.org/records/14182446
   - Download all files
   - Place them in: `data/zenodo/`

2. **Run the example:**
   ```bash
   python examples/load_zenodo_data.py
   ```

3. **Follow the prompts:**
   - The script will guide you through the process
   - You can choose to compare all models or use a single model

## Additional Examples

### Demo Script (Root Directory)

The repository also includes `demo.py` in the root directory, which demonstrates the system using synthetic EMG data:

```bash
python demo.py
```

This is useful for:
- Testing the installation
- Understanding the workflow without real data
- Quick prototyping

### Jupyter Notebooks

Interactive examples are available in the `notebooks/` directory:

- `example_usage.ipynb` - General usage with synthetic data
- `zenodo_dataset_example.ipynb` - Working with Zenodo dataset

To use the notebooks:
```bash
jupyter notebook notebooks/
```

## Creating Your Own Examples

You can create your own examples by following this pattern:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import ZenodoDataLoader
from src.pipeline import FatigueDetectionPipeline
from sklearn.model_selection import train_test_split

# 1. Load your data
loader = ZenodoDataLoader()
dataset = loader.load_dataset("data/your_data")

# 2. Prepare signals and labels
signals_list = [signals for signals, labels in dataset.values()]
labels_list = [labels for signals, labels in dataset.values()]

# 3. Initialize pipeline
pipeline = FatigueDetectionPipeline(sampling_rate=1000)

# 4. Prepare features
features_df, labels_array = pipeline.prepare_dataset(signals_list, labels_list)

# 5. Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels_array, test_size=0.2, random_state=42
)

pipeline.train(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Troubleshooting

### Data Not Found

If you see "DATA NOT FOUND" error:
1. Ensure files are in the correct directory (`data/zenodo/`)
2. Check that files are in CSV format
3. Verify file permissions

### Import Errors

If you see import errors:
1. Make sure you're running from the repository root
2. Check that all dependencies are installed: `pip install -r requirements.txt`
3. Verify Python version (3.7 or higher required)

### Memory Issues

For large datasets:
1. Process files one at a time
2. Reduce window size in preprocessing
3. Use smaller overlap in segmentation

## Documentation

For more information, see:
- [Main README](../README.md) - Project overview and installation
- [Zenodo Data Guide](../data/ZENODO_DATA.md) - Detailed Zenodo dataset documentation
- [Implementation Details](../IMPLEMENTATION.md) - Technical implementation

## Support

If you encounter issues:
1. Check the documentation above
2. Review the example code
3. Open an issue on GitHub: https://github.com/k2pitel/Muscle-fatigue-tracker

## License

MIT License - See the repository root for details.
