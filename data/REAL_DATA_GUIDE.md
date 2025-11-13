# Working with Real EMG Data

This guide explains how to work with the sEMG dataset you've added to the project.

## Current Status

You have 12 subject zip files in `data/sEMG_data/`:
- `subject_1` through `subject_13` (note: subject_9 is missing)
- Currently extracted: `subject_1` and `subject_10`

## Quick Start - Testing with Real Data

### Option 1: Use the Testing Notebook (Recommended)

Open and run `notebooks/real_data_testing.ipynb`:

```bash
# Open in VS Code or Jupyter
jupyter notebook notebooks/real_data_testing.ipynb
```

This notebook will:
- Load extracted subject data
- Visualize raw EMG signals from multiple muscles
- Extract features and train models
- Compare performance across different muscle groups
- Save trained models

### Option 2: Use the Command Line Tool

Check what's available:
```bash
python examples/extract_subject_data.py status
```

Extract a specific subject:
```bash
# Extract subject 2
python examples/extract_subject_data.py extract --subject 2

# Or use full name
python examples/extract_subject_data.py extract --subject subject_2
```

Extract all subjects (if you have disk space):
```bash
python examples/extract_subject_data.py extract-all
```

Clean up to free space:
```bash
python examples/extract_subject_data.py cleanup --subject 2
```

## Data Structure

Each subject folder contains:

```
subject_X/
├── trial_1.csv
├── trial_2.csv
├── ...
├── trial_12.csv
└── MVC/
    ├── subject_X_MVC_R_BICEPS.csv
    ├── subject_X_MVC_R_DELTOID_ANTERIOR.csv
    └── ...
```

### Trial Files

Each trial CSV contains:
- **Time column**: `X [s]` - timestamp in seconds
- **EMG channels**: Multiple muscle recordings
  - `R BICEPS BRACHII: EMG 1 [V]` - Right biceps
  - `R DELTOID ANTERIOR: EMG 2 [V]` - Right anterior deltoid
  - `R DELTOID MEDIUS: EMG 6 [V]` - Right medial deltoid
  - `R DELTOID POSTERIOR: EMG 7 [V]` - Right posterior deltoid

### Data Characteristics

- **Sampling rate**: ~1260 Hz (calculated from timestamps)
- **Trial duration**: ~95 seconds per trial
- **Number of trials**: 12 per subject
- **Signal format**: Voltage (V)

## Fatigue Labeling Strategy

The testing notebook uses this assumption:
- **Trials 1-6**: Non-fatigued (label = 0)
- **Trials 7-12**: Fatigued (label = 1)

This assumes a progressive fatigue protocol where subjects get increasingly fatigued across trials.

## Example Code Snippets

### Load a Subject's Data

```python
from src.preprocessing import EMGPreprocessor
import pandas as pd
import os

# Load trial 1 from subject 1
trial_1 = pd.read_csv('data/sEMG_data/subject_1/trial_1.csv')

# Extract biceps EMG signal
biceps_signal = trial_1['R BICEPS BRACHII: EMG 1 [V]'].values

# Get sampling rate
time = trial_1['X [s]'].values
sampling_rate = int(1 / (time[1] - time[0]))
print(f"Sampling rate: {sampling_rate} Hz")
```

### Preprocess and Extract Features

```python
from src.pipeline import FatigueDetectionPipeline

# Create pipeline
pipeline = FatigueDetectionPipeline(sampling_rate=1260, model_type='svm')

# Prepare multiple signals
signals = [biceps_signal_trial_1, biceps_signal_trial_2, ...]
labels = [0, 0, ...]  # 0 = non-fatigued, 1 = fatigued

# Extract features
features_df, labels_array = pipeline.prepare_dataset(
    signals, labels, 
    window_size=2520,  # 2 seconds at 1260 Hz
    overlap=0.5
)
```

### Train Model

```python
from sklearn.model_selection import train_test_split
from src.models import FatigueClassifier

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels_array, test_size=0.2, random_state=42
)

# Train
classifier = FatigueClassifier(model_type='svm')
classifier.train(X_train, y_train)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Tips for Working with Limited Disk Space

If you're running low on disk space:

1. **Work with one subject at a time**:
   ```bash
   # Extract subject
   python examples/extract_subject_data.py extract --subject 2
   
   # Do your analysis
   # ...
   
   # Clean up when done
   python examples/extract_subject_data.py cleanup --subject 2
   ```

2. **Read directly from zip files** (slower but saves space):
   ```python
   import zipfile
   import io
   import pandas as pd
   
   with zipfile.ZipFile('data/sEMG_data/subject_2.zip') as z:
       with z.open('subject_2/trial_1.csv') as f:
           df = pd.read_csv(io.TextIOWrapper(f))
   ```

3. **Process in batches**: Load, process, extract features, then discard raw data

## Next Steps

1. **Run the testing notebook**: `notebooks/real_data_testing.ipynb`
2. **Compare subjects**: Extract and analyze multiple subjects
3. **Multi-muscle analysis**: Combine features from all 4 muscle channels
4. **Cross-subject validation**: Train on some subjects, test on others
5. **Temporal analysis**: Track how features change across trials

## Troubleshooting

**Issue**: Out of disk space when extracting
- **Solution**: Extract subjects one at a time, or read directly from zip files

**Issue**: Import errors in notebook
- **Solution**: Make sure you're running from the notebooks directory and the path setup is correct

**Issue**: Different sampling rates across subjects
- **Solution**: Check the time column for each subject and calculate the actual sampling rate

## References

For more information about the dataset, see:
- `data/ZENODO_DATA.md` - Dataset documentation
- `data/README.md` - Data format guidelines
