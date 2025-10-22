# Data Directory

This directory is for storing EMG datasets used for muscle fatigue detection.

## Recommended Dataset

The project is designed to work with open EMG datasets with fatigue labels, such as:

**Cerqueira et al. (2024)** - Open EMG dataset with fatigue labels for muscle fatigue detection research.

## Data Format

The preprocessing pipeline expects data in the following format:

### CSV Format
- Each row should represent a sample or time point
- Columns should include:
  - EMG signal data (one or more channels)
  - Fatigue label (0 for non-fatigued, 1 for fatigued, or custom labels)
  - Optional: timestamp, subject ID, exercise type, etc.

### Example Structure

```csv
timestamp,emg_channel_1,emg_channel_2,fatigue_label,subject_id
0.001,0.123,-0.045,0,1
0.002,0.156,-0.032,0,1
0.003,0.187,-0.021,0,1
...
```

## Using Your Own Data

1. Place your EMG data files in this directory
2. Ensure data is in CSV format or convert it
3. Update the data loading code in your scripts:

```python
from src.preprocessing import EMGPreprocessor
import pandas as pd

preprocessor = EMGPreprocessor(sampling_rate=1000)  # Adjust to your sampling rate
data = preprocessor.load_data('data/your_dataset.csv')

# Extract signals and labels
signals = data['emg_column'].values  # Adjust column name
labels = data['fatigue_label'].values
```

## Data Preprocessing Notes

- **Sampling Rate**: Make sure to specify the correct sampling rate for your EMG device
  - Typical range: 1000-2000 Hz for surface EMG
  
- **Filtering**: The default bandpass filter (20-450 Hz) works for most surface EMG
  - Adjust if needed based on your signal characteristics
  
- **Segmentation**: Window size and overlap can be tuned based on:
  - Your sampling rate
  - Duration of muscle contractions
  - Desired temporal resolution

## Privacy and Ethics

- Do not commit actual patient/subject data to version control
- Ensure you have proper permissions to use any datasets
- Follow ethical guidelines for human subjects research
- Anonymize any personal information

## Synthetic Data

For testing without real data, use the demo script which generates synthetic EMG signals:

```bash
python demo.py
```

This creates realistic-looking EMG patterns for both fatigued and non-fatigued states.
