"""
Demo script for muscle fatigue detection system.
Generates synthetic EMG data and demonstrates the complete pipeline.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import EMGPreprocessor
from src.feature_extraction import EMGFeatureExtractor
from src.models import FatigueClassifier
from src.pipeline import FatigueDetectionPipeline, train_multiple_models
from sklearn.model_selection import train_test_split


def generate_synthetic_emg_data(n_samples=100, signal_length=5000, sampling_rate=1000):
    """
    Generate synthetic EMG data for demonstration.
    Simulates non-fatigued and fatigued muscle signals.
    
    Args:
        n_samples (int): Number of samples to generate
        signal_length (int): Length of each signal
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        tuple: (signals, labels)
    """
    signals = []
    labels = []
    
    for i in range(n_samples):
        # Determine if fatigued or not (0: non-fatigued, 1: fatigued)
        is_fatigued = i % 2
        
        # Time array
        t = np.arange(signal_length) / sampling_rate
        
        if is_fatigued:
            # Fatigued signal: lower frequency, higher amplitude
            freq = np.random.uniform(15, 30)  # Lower frequency
            amplitude = np.random.uniform(0.8, 1.5)  # Higher amplitude
            noise_level = 0.3
        else:
            # Non-fatigued signal: higher frequency, lower amplitude
            freq = np.random.uniform(50, 100)  # Higher frequency
            amplitude = np.random.uniform(0.3, 0.7)  # Lower amplitude
            noise_level = 0.2
        
        # Generate signal with multiple frequency components
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        signal += 0.3 * amplitude * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.2 * amplitude * np.sin(2 * np.pi * freq * 3 * t)
        
        # Add noise
        noise = np.random.normal(0, noise_level, signal_length)
        signal += noise
        
        signals.append(signal)
        labels.append(is_fatigued)
    
    return signals, labels


def main():
    """
    Main demo function.
    """
    print("=" * 60)
    print("Muscle Fatigue Detection System - Demo")
    print("=" * 60)
    
    # Configuration
    sampling_rate = 1000
    n_samples = 100
    signal_length = 5000
    window_size = 1000
    overlap = 0.5
    
    print("\n1. Generating synthetic EMG data...")
    signals, labels = generate_synthetic_emg_data(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=sampling_rate
    )
    print(f"   Generated {n_samples} signals")
    print(f"   Signal length: {signal_length} samples ({signal_length/sampling_rate} seconds)")
    print(f"   Non-fatigued samples: {labels.count(0)}")
    print(f"   Fatigued samples: {labels.count(1)}")
    
    # Create pipelines for different models
    model_types = ['knn', 'svm', 'logistic']
    
    print("\n2. Processing signals and extracting features...")
    # Use KNN pipeline for feature extraction
    pipeline = FatigueDetectionPipeline(
        sampling_rate=sampling_rate,
        model_type='knn'
    )
    
    # Prepare dataset
    features_df, labels_array = pipeline.prepare_dataset(
        signals, labels, window_size=window_size, overlap=overlap
    )
    
    print(f"   Extracted {len(features_df)} feature vectors")
    print(f"   Features: {list(features_df.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_array, test_size=0.2, random_state=42, stratify=labels_array
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    print("\n3. Training and evaluating models...")
    print("-" * 60)
    
    results = {}
    
    for model_type in model_types:
        print(f"\n   Training {model_type.upper()} model...")
        
        # Create classifier
        classifier = FatigueClassifier(model_type=model_type)
        
        # Train
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics
        }
        
        print(f"   {model_type.upper()} Accuracy: {metrics['accuracy']:.4f}")
        
        # Print classification report
        report = metrics['classification_report']
        print(f"   Precision (Non-fatigued): {report['0']['precision']:.4f}")
        print(f"   Recall (Non-fatigued): {report['0']['recall']:.4f}")
        print(f"   Precision (Fatigued): {report['1']['precision']:.4f}")
        print(f"   Recall (Fatigued): {report['1']['recall']:.4f}")
    
    print("\n" + "-" * 60)
    print("\n4. Model Comparison:")
    print("-" * 60)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    print(f"   Best Model: {best_model[0].upper()}")
    print(f"   Best Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
    
    print("\n5. Saving best model...")
    model_path = os.path.join('models', f'best_model_{best_model[0]}.joblib')
    os.makedirs('models', exist_ok=True)
    best_model[1]['classifier'].save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n6. Testing prediction on new signal...")
    # Generate a test signal
    test_signals, test_labels = generate_synthetic_emg_data(n_samples=1, signal_length=signal_length)
    test_signal = test_signals[0]
    true_label = test_labels[0]
    
    # Create pipeline with best model
    test_pipeline = FatigueDetectionPipeline(
        sampling_rate=sampling_rate,
        model_type=best_model[0]
    )
    test_pipeline.classifier = best_model[1]['classifier']
    
    # Make prediction
    predictions = test_pipeline.predict(test_signal, window_size=window_size, overlap=overlap)
    predicted_label = int(np.round(np.mean(predictions)))
    
    print(f"   True label: {'Fatigued' if true_label == 1 else 'Non-fatigued'} ({true_label})")
    print(f"   Predicted label: {'Fatigued' if predicted_label == 1 else 'Non-fatigued'} ({predicted_label})")
    print(f"   Prediction segments: {predictions}")
    print(f"   Average prediction: {np.mean(predictions):.2f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Replace synthetic data with real EMG dataset (Cerqueira et al., 2024)")
    print("- Adjust preprocessing parameters for your specific data")
    print("- Fine-tune model hyperparameters")
    print("- Implement cross-validation for robust evaluation")
    print("- Use visualization module to analyze results")
    print("=" * 60)


if __name__ == "__main__":
    main()
