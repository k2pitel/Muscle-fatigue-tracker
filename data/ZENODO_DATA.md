# Zenodo EMG Dataset - Record 14182446

This document provides information about the EMG dataset from Zenodo record 14182446 and how to use it with the Muscle Fatigue Tracker.

## About the Dataset

**Zenodo Record:** https://zenodo.org/records/14182446

This Zenodo record contains EMG (Electromyography) data for muscle fatigue detection research. The dataset is designed to work with machine learning algorithms to classify fatigue levels in muscles during physical activity.

## Downloading the Dataset

### Automatic Download

You can use the built-in data loader to automatically download the dataset:

```python
from src.data_loader import download_zenodo_dataset

# Download to data/zenodo directory
data_dir = download_zenodo_dataset(
    record_id="14182446",
    output_dir="data/zenodo"
)
```

Or run the example script:

```bash
python examples/load_zenodo_data.py
```

### Manual Download

If automatic download is not available in your environment:

1. Visit: https://zenodo.org/records/14182446
2. Download all files from the record
3. Place them in the `data/zenodo/` directory in this repository

## Dataset Structure

The dataset typically includes:

- **EMG signal files**: Raw or preprocessed EMG signals
- **Label files**: Fatigue classifications (e.g., 0 = non-fatigued, 1 = fatigued)
- **Metadata**: Information about recording conditions, subjects, exercises
- **Documentation**: Description of the dataset structure and collection methods

### Expected File Formats

The data loader supports multiple formats:

- **CSV files**: Most common format for EMG data
  - Columns: signal values, timestamps, labels
  - One file per recording session or subject
  
- **MAT files**: MATLAB format (if scipy is available)
  - Structured arrays with signal and metadata

- **TXT files**: Plain text format
  - Tab or comma separated values

## Using the Dataset

### Quick Start

```python
from src.data_loader import ZenodoDataLoader
from src.pipeline import FatigueDetectionPipeline
from sklearn.model_selection import train_test_split

# 1. Load the data
loader = ZenodoDataLoader("14182446")
dataset = loader.load_dataset("data/zenodo")

# 2. Prepare for training
signals_list = []
labels_list = []

for filename, (signals, labels) in dataset.items():
    signals_list.append(signals)
    if labels is not None:
        labels_list.append(labels)

# 3. Train a model
pipeline = FatigueDetectionPipeline(sampling_rate=1000)
features_df, labels_array = pipeline.prepare_dataset(signals_list, labels_list)

X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels_array, test_size=0.2, random_state=42
)

pipeline.train(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Complete Example

See `examples/load_zenodo_data.py` for a complete workflow that includes:
- Data downloading
- Loading and validation
- Preprocessing
- Feature extraction
- Model training and evaluation
- Model comparison

Run it with:
```bash
python examples/load_zenodo_data.py
```

## Data Preprocessing

The Zenodo dataset may require preprocessing depending on its format:

### 1. Signal Filtering

EMG signals typically need bandpass filtering (20-450 Hz):

```python
from src.preprocessing import EMGPreprocessor

preprocessor = EMGPreprocessor(sampling_rate=1000)
filtered_signal = preprocessor.bandpass_filter(raw_signal)
```

### 2. Normalization

Normalize signals to zero mean and unit variance:

```python
normalized_signal = preprocessor.normalize(filtered_signal)
```

### 3. Segmentation

Segment signals into windows for feature extraction:

```python
windows = preprocessor.segment_signal(
    signal, 
    window_size=1000,  # 1 second windows at 1000 Hz
    overlap=0.5        # 50% overlap
)
```

## Feature Extraction

Extract features from the EMG signals:

```python
from src.feature_extraction import EMGFeatureExtractor

extractor = EMGFeatureExtractor(sampling_rate=1000)
features = extractor.extract_all_features(signal)

# Features include:
# - RMS (Root Mean Square)
# - MAV (Mean Absolute Value)
# - Variance
# - Waveform Length
# - Zero Crossing Rate
# - Median Frequency
# - Mean Frequency
# - Spectral Entropy
```

## Sampling Rate

The sampling rate is crucial for proper signal processing. Common EMG sampling rates:

- **1000 Hz**: Standard for surface EMG
- **2000 Hz**: High-quality recordings
- **500 Hz**: Minimum for fatigue analysis

Check the dataset documentation or metadata to determine the correct sampling rate, then configure the pipeline accordingly:

```python
pipeline = FatigueDetectionPipeline(sampling_rate=YOUR_SAMPLING_RATE)
```

## Fatigue Labels

The dataset should include fatigue labels for supervised learning:

- **Binary classification**: 0 = non-fatigued, 1 = fatigued
- **Multi-class**: Different fatigue levels (e.g., 0, 1, 2, 3)

The labels may be:
- In a separate column in the same CSV file
- In a separate label file
- Encoded in the filename

The data loader will attempt to automatically detect and load labels.

## Model Training

### Single Model

```python
from src.pipeline import FatigueDetectionPipeline

pipeline = FatigueDetectionPipeline(sampling_rate=1000, model_type='svm')
pipeline.train(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
```

### Multiple Model Comparison

```python
from src.pipeline import train_multiple_models

results = train_multiple_models(X_train, y_train, X_test, y_test)

for model_name, result in results.items():
    print(f"{model_name}: Accuracy = {result['metrics']['accuracy']:.4f}")
```

## Citation

If you use this Zenodo dataset in your research, please cite:

```
@dataset{zenodo14182446,
  title={[Dataset Title]},
  author={[Authors]},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.14182446},
  url={https://zenodo.org/records/14182446}
}
```

Visit the Zenodo record page for the complete citation information.

## Troubleshooting

### Data Not Loading

If you encounter issues loading the data:

1. **Check file format**: Ensure files are in CSV, MAT, or TXT format
2. **Check column names**: The loader looks for columns named 'emg', 'signal', or numeric columns
3. **Check file encoding**: Try opening files with different encodings (utf-8, latin-1)
4. **Verify data directory**: Ensure files are in `data/zenodo/`

### Sampling Rate Issues

If feature extraction fails:

1. Verify the sampling rate matches your data
2. Check signal length (should be long enough for windowing)
3. Ensure signals are numeric arrays, not strings

### Memory Issues

For large datasets:

1. Process files one at a time instead of loading all at once
2. Use smaller window sizes
3. Reduce overlap in segmentation
4. Use batch processing

## Additional Resources

- **Main README**: See [../README.md](../README.md) for general usage
- **Example Script**: Run `python examples/load_zenodo_data.py`
- **Demo Script**: Run `python demo.py` for synthetic data example
- **Jupyter Notebooks**: See `notebooks/` for interactive examples

## Support

For questions or issues:

1. Check the dataset documentation on Zenodo
2. Review the example scripts in this repository
3. Open an issue on GitHub: https://github.com/k2pitel/Muscle-fatigue-tracker

## License

The Zenodo dataset has its own license - check the record page for details.
This code is available under the MIT License.
