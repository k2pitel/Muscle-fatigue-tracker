# Muscle Fatigue Tracker

A machine learning-based system for detecting muscle fatigue using Electromyography (EMG) signals. This project implements various ML algorithms (KNN, SVM, Logistic Regression) to classify muscle fatigue levels, helping optimize strength training and improve training efficiency.

## Overview

This project explores using machine learning to detect muscle fatigue and optimize strength training. Using EMG data with fatigue labels (designed for datasets like Cerqueira et al., 2024), the system:

- **Preprocesses** EMG signals (filtering, normalization, segmentation)
- **Extracts features** such as RMS, mean/median frequency, spectral entropy
- **Trains models** using KNN, SVM, and Logistic Regression
- **Classifies** fatigue levels to improve training efficiency
- **Demonstrates** ML's potential in health technology applications

## Features

### Signal Preprocessing
- Bandpass filtering (20-450 Hz) to remove noise
- Signal normalization
- Windowing and segmentation with configurable overlap

### Feature Extraction
- **Time-domain features:**
  - Root Mean Square (RMS)
  - Mean Absolute Value (MAV)
  - Variance
  - Waveform Length
  - Zero Crossing Rate

- **Frequency-domain features:**
  - Median Frequency
  - Mean Frequency
  - Spectral Entropy

### Machine Learning Models
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Hyperparameter tuning with GridSearchCV
- Cross-validation support

## Project Structure

```
Muscle-fatigue-tracker/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # EMG signal preprocessing
│   ├── feature_extraction.py   # Feature extraction from EMG signals
│   ├── models.py               # ML classifiers (KNN, SVM, Logistic Regression)
│   ├── pipeline.py             # Complete training pipeline
│   └── visualization.py        # Plotting and visualization tools
├── data/                        # Place your EMG datasets here
├── models/                      # Saved trained models
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Unit tests
├── demo.py                      # Demo script with synthetic data
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/k2pitel/Muscle-fatigue-tracker.git
cd Muscle-fatigue-tracker
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

## Quick Start

Run the demo script to see the system in action with synthetic data:

```bash
python demo.py
```

This will:
1. Generate synthetic EMG data (non-fatigued and fatigued signals)
2. Preprocess and extract features
3. Train KNN, SVM, and Logistic Regression models
4. Evaluate and compare model performance
5. Save the best model
6. Demonstrate prediction on a new signal

## Usage

### Basic Usage

```python
from src.pipeline import FatigueDetectionPipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize pipeline with desired model
pipeline = FatigueDetectionPipeline(sampling_rate=1000, model_type='knn')

# Prepare your data (signals: list of EMG arrays, labels: fatigue levels)
features_df, labels_array = pipeline.prepare_dataset(signals, labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels_array, test_size=0.2, random_state=42
)

# Train the model
pipeline.train(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Make predictions on new signals
predictions = pipeline.predict(new_signal)
```

### Using Individual Components

```python
from src.preprocessing import EMGPreprocessor
from src.feature_extraction import EMGFeatureExtractor
from src.models import FatigueClassifier

# Preprocess signal
preprocessor = EMGPreprocessor(sampling_rate=1000)
processed_signal = preprocessor.preprocess_pipeline(raw_signal)

# Extract features
extractor = EMGFeatureExtractor(sampling_rate=1000)
features = extractor.extract_all_features(processed_signal)

# Train classifier
classifier = FatigueClassifier(model_type='svm')
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)
```

### Comparing Multiple Models

```python
from src.pipeline import train_multiple_models

# Train and compare all models
results = train_multiple_models(X_train, y_train, X_test, y_test)

# Access results
for model_name, result in results.items():
    print(f"{model_name}: {result['metrics']['accuracy']:.4f}")
```

## Working with Real EMG Data

To use real EMG data (e.g., Cerqueira et al., 2024 dataset):

1. Place your EMG data files in the `data/` directory
2. Ensure data format: CSV files with EMG signal columns and fatigue labels
3. Load and process:

```python
from src.preprocessing import EMGPreprocessor
import pandas as pd

# Load data
preprocessor = EMGPreprocessor(sampling_rate=your_sampling_rate)
data = preprocessor.load_data('data/your_emg_data.csv')

# Extract signals and labels
signals = data['emg_column'].values  # Adjust column names
labels = data['fatigue_label'].values
```

## Model Performance

The system evaluates models using:
- Accuracy
- Precision and Recall per class
- Confusion Matrix
- Classification Report
- Cross-validation scores

## Visualization

Use the visualization module to analyze results:

```python
from src.visualization import (
    plot_emg_signal,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_distributions
)

# Plot EMG signal
plot_emg_signal(signal_data, sampling_rate=1000)

# Plot confusion matrix
plot_confusion_matrix(confusion_matrix, class_names=['Non-fatigued', 'Fatigued'])

# Compare models
plot_model_comparison(results)
```

## Configuration

Key parameters to adjust:

- **Sampling Rate**: Set according to your EMG device (typically 1000-2000 Hz)
- **Window Size**: Size of signal segments for feature extraction (default: 1000 samples)
- **Overlap**: Overlap between windows (default: 0.5 or 50%)
- **Filter Parameters**: Bandpass filter cutoff frequencies (default: 20-450 Hz)

## Future Enhancements

- [ ] Deep learning models (CNN, LSTM) for temporal pattern recognition
- [ ] Real-time fatigue detection
- [ ] Multi-channel EMG support
- [ ] Integration with wearable EMG devices
- [ ] Web interface for visualization
- [ ] Mobile app for athletes

## References

This project is designed to work with EMG datasets such as:
- Cerqueira, M. S., et al. (2024). Open EMG dataset with fatigue labels for muscle fatigue detection research.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@software{muscle_fatigue_tracker,
  title={Muscle Fatigue Tracker: ML-based EMG Fatigue Detection},
  author={k2pitel},
  year={2025},
  url={https://github.com/k2pitel/Muscle-fatigue-tracker}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.