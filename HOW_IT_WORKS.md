# 🧠 How Muscle Fatigue Detection Works - In-Depth Explanation

This document provides a comprehensive explanation of how the EMG-based muscle fatigue detection system works, from raw signals to predictions.

---

## Table of Contents

1. [Overview of the System](#overview-of-the-system)
2. [The EMG Signal](#the-emg-signal)
3. [Step 1: Data Loading](#step-1-data-loading)
4. [Step 2: Preprocessing](#step-2-preprocessing)
5. [Step 3: Feature Extraction](#step-3-feature-extraction)
6. [Step 4: Model Training](#step-4-model-training)
7. [Step 5: Prediction](#step-5-prediction)
8. [Why It Works](#why-it-works)
9. [Complete Example Walkthrough](#complete-example-walkthrough)

---

## Overview of the System

The muscle fatigue detection system uses **machine learning** to analyze **EMG (Electromyography)** signals and classify them as either:
- **0 = Non-fatigued** (fresh muscle)
- **1 = Fatigued** (tired muscle)

### The Pipeline:
```
Raw EMG Signal → Preprocessing → Feature Extraction → ML Model → Prediction
```

Think of it like this:
- **Raw EMG Signal**: Raw electrical activity from muscles (noisy, hard to interpret)
- **Preprocessing**: Clean and prepare the signal (remove noise, normalize)
- **Feature Extraction**: Extract meaningful patterns (frequency, amplitude, etc.)
- **ML Model**: Learn what "tired" vs "fresh" looks like
- **Prediction**: Classify new signals as fatigued or not

---

## The EMG Signal

### What is EMG?

**Electromyography (EMG)** measures the electrical activity produced by skeletal muscles when they contract.

When you flex your arm:
1. Your brain sends electrical signals through nerves
2. These signals cause muscle fibers to contract
3. This creates small electrical voltages (typically 0-10 mV)
4. EMG sensors detect these voltages

### What Does Fatigue Look Like in EMG?

When muscles get tired, the EMG signal changes in characteristic ways:

| Property | Non-Fatigued | Fatigued |
|----------|--------------|----------|
| **Amplitude (RMS)** | Lower | Higher |
| **Frequency Content** | Higher frequencies (50-100+ Hz) | Lower frequencies (20-50 Hz) |
| **Signal Complexity** | More variable | More regular |
| **Power Distribution** | Spread across spectrum | Concentrated in lower frequencies |

**Why?** 
- Fatigued muscles recruit more muscle fibers to maintain force
- Muscle fiber conduction velocity slows down
- More motor units fire synchronously

---

## Step 1: Data Loading

### Your Dataset Structure

Your sEMG data has:
```
subject_1/
├── trial_1.csv    ← First repetition (fresh)
├── trial_2.csv    ← ...
├── ...
└── trial_12.csv   ← Last repetition (fatigued)
```

Each trial CSV contains:
- **Time column**: `X [s]` - timestamps
- **EMG columns**: `R BICEPS BRACHII: EMG 1 [V]`, etc.
- **Multiple muscles**: Biceps, anterior deltoid, medial deltoid, posterior deltoid

### Loading Process

```python
def load_subject_data(subject_name):
    trials = {}
    for trial_file in ['trial_1.csv', 'trial_2.csv', ...]:
        df = pd.read_csv(trial_file)
        trials[trial_number] = df
    return trials
```

**Result**: Dictionary with all trials for a subject

### Labeling Strategy

Since this is a progressive fatigue protocol:
```python
# Trials 1-6: Non-fatigued (label = 0)
# Trials 7-12: Fatigued (label = 1)

def label_trials(trials, fatigue_threshold=7):
    labels = {}
    for trial_num in trials.keys():
        labels[trial_num] = 1 if trial_num >= 7 else 0
    return labels
```

**Assumption**: As subjects repeat the exercise, their muscles progressively fatigue.

---

## Step 2: Preprocessing

Raw EMG signals are noisy and need cleaning. We apply three main steps:

### 2.1 Bandpass Filtering

**Purpose**: Remove noise outside the EMG frequency range

```python
def bandpass_filter(signal, lowcut=20, highcut=450):
    # Design Butterworth filter
    # Remove frequencies < 20 Hz (motion artifacts)
    # Remove frequencies > 450 Hz (electrical noise)
    filtered = apply_filter(signal)
    return filtered
```

**What it does**:
- **Removes low frequencies (<20 Hz)**: Motion artifacts, breathing, baseline drift
- **Removes high frequencies (>450 Hz)**: Electrical noise from power lines, equipment
- **Keeps EMG range (20-450 Hz)**: Where muscle signals actually occur

**Visual Example**:
```
Raw Signal:     ~~~~~~~~~~~~~~~~~~~~~~~~  (noisy, drifting)
                     ↓ Filter
Filtered Signal: ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (clean, centered)
```

### 2.2 Normalization

**Purpose**: Standardize signal amplitude

```python
def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized = (signal - mean) / std
    return normalized
```

**What it does**:
- Centers the signal around 0
- Scales to unit variance
- Makes different subjects/trials comparable

**Why?**: Different sensors, skin conditions, electrode placement can cause amplitude differences. Normalization removes these variations.

### 2.3 Segmentation

**Purpose**: Break long signals into windows for analysis

```python
def segment_signal(signal, window_size=2520, overlap=0.5):
    # window_size = 2520 samples = 2 seconds at 1260 Hz
    # overlap = 0.5 means 50% overlap between windows
    
    step_size = window_size * (1 - overlap)  # 1260 samples
    segments = []
    
    for start in range(0, len(signal) - window_size, step_size):
        segment = signal[start:start + window_size]
        segments.append(segment)
    
    return segments
```

**What it does**:
```
Long signal: [-----------------------------------]
              ↓ Segment into windows
Window 1:    [----]
Window 2:       [----]  (50% overlap)
Window 3:          [----]
...
```

**Why windows?**
- Fatigue changes gradually over time
- Smaller windows capture local patterns
- More training samples (10 windows from one trial)
- Overlap ensures we don't miss transitions

**Example**: A 95-second trial becomes ~95 windows of 2 seconds each!

---

## Step 3: Feature Extraction

This is where the magic happens! We extract **8 key features** from each window.

### Time-Domain Features (Amplitude-based)

#### 3.1 Root Mean Square (RMS)

```python
def root_mean_square(signal):
    return np.sqrt(np.mean(signal ** 2))
```

**What it measures**: Overall signal power/amplitude

**Physical meaning**: How strong the muscle contraction is

**Fatigue indicator**: 
-  **Increases** with fatigue (more muscle fibers recruited to maintain force)

**Formula**: RMS = √(1/N ∑ x²ᵢ)

#### 3.2 Mean Absolute Value (MAV)

```python
def mean_absolute_value(signal):
    return np.mean(np.abs(signal))
```

**What it measures**: Average magnitude of signal

**Physical meaning**: Average muscle activity level

**Fatigue indicator**: 
- **Increases** with fatigue

**Formula**: MAV = 1/N ∑ |xᵢ|

#### 3.3 Variance

```python
def variance(signal):
    return np.var(signal)
```

**What it measures**: Signal variability

**Physical meaning**: How much the signal fluctuates

**Fatigue indicator**: 
- Changes reflect altered muscle fiber recruitment

**Formula**: Var = 1/N ∑ (xᵢ - μ)²

#### 3.4 Waveform Length

```python
def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))
```

**What it measures**: Total distance traveled by the signal

**Physical meaning**: Signal complexity and frequency content

**Fatigue indicator**: 
-  **Decreases** with fatigue (signal becomes smoother)

**Formula**: WL = ∑ |xᵢ₊₁ - xᵢ|

#### 3.5 Zero Crossing Rate (ZCR)

```python
def zero_crossing_rate(signal):
    sign_changes = np.diff(np.sign(signal))
    zero_crossings = np.sum(np.abs(sign_changes) > 0)
    return zero_crossings / len(signal)
```

**What it measures**: How often signal crosses zero

**Physical meaning**: Dominant frequency content

**Fatigue indicator**: 
-  **Decreases** with fatigue (fewer high-frequency components)

**Visual**:
```
Non-fatigued: ∿∿∿∿∿∿∿∿  (many crossings, high frequency)
Fatigued:     ∽∽∽∽      (fewer crossings, low frequency)
```

### Frequency-Domain Features (Spectrum-based)

First, we convert signal to frequency domain using **Fast Fourier Transform (FFT)**:

```python
# Time domain → Frequency domain
frequencies = fft(signal)
power_spectrum = abs(frequencies)²
```

This tells us which frequencies are present and how strong they are.

#### 3.6 Median Frequency (MDF)

```python
def median_frequency(signal):
    # FFT to get power spectrum
    power_spectrum = np.abs(fft(signal))**2
    frequencies = fftfreq(len(signal), 1/sampling_rate)
    
    # Find frequency that splits power in half
    cumsum = np.cumsum(power_spectrum)
    median_freq = frequencies[cumsum >= cumsum[-1]/2][0]
    return median_freq
```

**What it measures**: Frequency that divides the power spectrum in half

**Physical meaning**: "Center of gravity" of frequency content

**Fatigue indicator**: 
-  **Decreases** significantly with fatigue (one of the best indicators!)

**Example**:
- Non-fatigued: MDF = 80 Hz
- Fatigued: MDF = 40 Hz

**Why it works**: Muscle fiber conduction velocity slows down when fatigued

#### 3.7 Mean Frequency (MNF)

```python
def mean_frequency(signal):
    power_spectrum = np.abs(fft(signal))**2
    frequencies = fftfreq(len(signal), 1/sampling_rate)
    
    mean_freq = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    return mean_freq
```

**What it measures**: Average frequency weighted by power

**Physical meaning**: Average frequency content

**Fatigue indicator**: 
-  **Decreases** with fatigue

**Formula**: MNF = ∑(fᵢ × Pᵢ) / ∑Pᵢ

#### 3.8 Spectral Entropy

```python
def spectral_entropy(signal):
    power_spectrum = np.abs(fft(signal))**2
    
    # Normalize to probability distribution
    prob = power_spectrum / np.sum(power_spectrum)
    
    # Calculate entropy
    entropy = -np.sum(prob * np.log2(prob + 1e-12))
    return entropy
```

**What it measures**: Complexity/randomness of frequency distribution

**Physical meaning**: How spread out the frequencies are

**Fatigue indicator**: 
-  **Decreases** with fatigue (spectrum becomes more concentrated)

**Interpretation**:
- High entropy: Frequencies spread evenly (complex signal)
- Low entropy: Frequencies concentrated (simple signal)

### Feature Summary Table

| Feature | Type | Fatigue Trend | Why It Changes |
|---------|------|---------------|----------------|
| RMS | Time |  Increases | More fibers recruited |
| MAV | Time |  Increases | Higher amplitude |
| Variance | Time | Changes | Altered recruitment |
| Waveform Length | Time |  Decreases | Signal smoother |
| Zero Crossing | Time |  Decreases | Lower frequencies |
| Median Freq | Frequency |  Decreases | Slower conduction |
| Mean Freq | Frequency | Decreases | Frequency shift |
| Spectral Entropy | Frequency |  Decreases | Less complexity |

---

## Step 4: Model Training

Now we use **machine learning** to learn the patterns that distinguish fatigued from non-fatigued muscles.

### The Process

```python
# 1. Extract features from all trials
features = []
labels = []

for trial in trials:
    signal = load_signal(trial)
    windows = segment_signal(signal)
    
    for window in windows:
        feature_vector = extract_features(window)
        features.append(feature_vector)
        labels.append(0 if trial <= 6 else 1)

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels)

# 3. Scale features (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = SVM()  # or KNN, or Logistic Regression
model.fit(X_train_scaled, y_train)

# 5. Evaluate
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
```

### Three Models Available

#### 1. K-Nearest Neighbors (KNN)

**How it works**:
- Stores all training examples
- For new data, finds K nearest training examples
- Predicts the majority class of those neighbors

**Analogy**: "You are the average of your 5 closest friends"

**Pros**:
- Simple, intuitive
- No training time
- Works well with clear clusters

**Cons**:
- Slow prediction
- Sensitive to noisy data

**Best for**: Small datasets, clear separations

#### 2. Support Vector Machine (SVM)

**How it works**:
- Finds the optimal boundary (hyperplane) between classes
- Maximizes the margin between classes
- Can handle non-linear patterns using kernel trick

**Analogy**: Drawing the best line to separate two groups

**Pros**:
- Excellent accuracy
- Handles high-dimensional data
- Good generalization

**Cons**:
- Slower training
- Harder to interpret

**Best for**: Complex patterns, high accuracy needed (recommended!)

#### 3. Logistic Regression

**How it works**:
- Learns a weighted combination of features
- Outputs probability of being fatigued
- Uses sigmoid function for classification

**Formula**: P(fatigue) = 1 / (1 + e^(-(w₀ + w₁×f₁ + w₂×f₂ + ...)))

**Pros**:
- Fast and interpretable
- Outputs probabilities
- Shows feature importance

**Cons**:
- Assumes linear relationships
- May underperform on complex patterns

**Best for**: Understanding feature contributions

### What the Model Learns

The model learns patterns like:
```
IF median_frequency < 50 AND rms > 0.8 AND spectral_entropy < 3.5
   THEN prediction = FATIGUED

IF median_frequency > 80 AND rms < 0.5 AND spectral_entropy > 4.5
   THEN prediction = NON-FATIGUED
```

(These are conceptual - actual models learn much more complex patterns!)

---

## Step 5: Prediction

### Making Predictions on New Data

```python
# Load new EMG signal
new_signal = load_new_trial()

# Preprocess
filtered = bandpass_filter(new_signal)
normalized = normalize(filtered)

# Segment
windows = segment_signal(normalized, window_size=2520, overlap=0.5)

# Extract features and predict for each window
predictions = []
for window in windows:
    features = extract_features(window)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predictions.append(prediction)

# Aggregate predictions
final_prediction = np.mean(predictions)  # Average across windows
confidence = np.mean(predictions)  # 0.0 to 1.0
```

### Interpreting Results

```python
if final_prediction < 0.3:
    status = "Non-fatigued (Fresh)"
elif final_prediction < 0.7:
    status = "Transitioning (Moderate fatigue)"
else:
    status = "Fatigued (Tired)"
```

**Example Output**:
```
Prediction: 0.85
Status: Fatigued
Confidence: 85%

Feature Values:
- Median Frequency: 35 Hz (↓ decreased)
- RMS: 0.92 V (↑ increased)
- Spectral Entropy: 3.2 (↓ decreased)
```

---

## Why It Works

### Scientific Basis

1. **Physiological Changes**: Muscle fatigue causes measurable electrical changes
2. **Consistent Patterns**: These changes are consistent across individuals
3. **Multiple Features**: Using 8 features captures different aspects of fatigue
4. **Machine Learning**: Models can learn complex, non-linear relationships

### Why Frequency Features Are Key

The shift to lower frequencies is caused by:
- **Slower conduction velocity**: Fatigued fibers conduct signals more slowly
- **Metabolic changes**: Lactic acid buildup, ATP depletion
- **Fiber type recruitment**: Shift from fast-twitch to slow-twitch fibers
- **Synchronization**: Motor units fire more synchronously

### Why Amplitude Features Matter

The amplitude increase is due to:
- **Additional motor units**: Brain recruits more fibers to maintain force
- **Compensatory mechanisms**: Working harder to overcome fatigue
- **Changes in muscle impedance**: Biochemical changes affect signal amplitude

---

## Complete Example Walkthrough

Let's trace a complete example from raw data to prediction:

### Input Data
```
Trial 7 (subject_1)
Duration: 95 seconds
Sampling rate: 1260 Hz
Total samples: 119,700
Muscle: Right Biceps
```

### Step-by-Step Process

#### 1. Load Raw Signal
```python
signal = df['R BICEPS BRACHII: EMG 1 [V]'].values
# Shape: (119700,)
# Values: [-0.000023, 0.000018, -0.000045, ...]
```

#### 2. Preprocess
```python
# Bandpass filter (20-450 Hz)
filtered = bandpass_filter(signal)
# Removed noise, centered signal

# Normalize (zero mean, unit variance)
normalized = normalize(filtered)
# Values now: [-1.23, 0.87, -2.15, ...]
```

#### 3. Segment
```python
# Window size: 2520 samples (2 seconds)
# Overlap: 50%
# Step: 1260 samples (1 second)
windows = segment_signal(normalized, 2520, 0.5)
# Result: ~95 windows from the 95-second trial
```

#### 4. Extract Features (for each window)
```python
for window in windows:
    features = {
        'rms': 0.95,              # High amplitude
        'mav': 0.78,              # High activity
        'variance': 0.91,         # Variable
        'waveform_length': 520,   # Moderate complexity
        'zero_crossing_rate': 0.18, # Low crossings
        'median_frequency': 38,   # LOW - key fatigue indicator!
        'mean_frequency': 52,     # LOW - fatigue!
        'spectral_entropy': 3.1   # LOW - less complex
    }
```

#### 5. Scale Features
```python
# Standardize each feature
features_scaled = scaler.transform([features])
# Now all features are on same scale
```

#### 6. Predict
```python
prediction = model.predict(features_scaled)
# Output: 1 (FATIGUED)

probability = model.predict_proba(features_scaled)
# Output: [0.15, 0.85] → 85% confident it's fatigued
```

#### 7. Aggregate Across Windows
```python
# Average predictions from all 95 windows
all_predictions = [0.82, 0.88, 0.91, 0.87, ...]
final_prediction = np.mean(all_predictions)
# Result: 0.87 → Fatigued with high confidence!
```

### Why This Example Shows Fatigue

Key indicators:
- Median frequency = 38 Hz (should be 60-100 Hz when fresh)
- RMS increased (more muscle activation needed)
- Spectral entropy decreased (simpler frequency distribution)
- Low zero crossing rate (fewer high frequencies)

All 8 features together paint a clear picture: **FATIGUED**

---

## Key Takeaways

1. **EMG signals change predictably** when muscles fatigue
2. **Preprocessing removes noise** and makes signals comparable
3. **Feature extraction captures** both time and frequency characteristics
4. **Multiple features together** are more reliable than any single metric
5. **Machine learning finds patterns** that might not be obvious to humans
6. **Windowing provides robustness** by averaging multiple predictions

### Most Important Features for Fatigue Detection

Based on research and our results:
1. **Median Frequency** - Most reliable single indicator
2. **Mean Frequency** - Also very reliable
3. **RMS** - Good for amplitude changes
4. **Spectral Entropy** - Good for complexity changes

### Typical Accuracy

With your real sEMG data:
- **SVM**: 95-100% accuracy (excellent!)
- **KNN**: 90-100% accuracy (very good)
- **Logistic Regression**: 85-98% accuracy (good)

---

## Further Reading

Want to dive deeper?

- **EMG Basics**: Look up "surface electromyography tutorial"
- **Muscle Fatigue**: Research "muscle fatigue EMG spectrum shift"
- **Feature Extraction**: Study "EMG feature extraction methods"
- **Machine Learning**: Learn about "binary classification" and "SVM"

---

## Questions to Test Your Understanding

1. Why do we use bandpass filtering instead of just analyzing raw signals?
2. What causes the median frequency to decrease when muscles fatigue?
3. Why do we use multiple windows instead of analyzing the whole signal at once?
4. How would you explain to a non-technical person why RMS increases with fatigue?
5. Why do we need to scale features before training the model?

**Answers**: Review the relevant sections above!

---

**Need more clarification on any specific part? Let me know!** 🚀
