"""
Example script demonstrating how to load and use EMG data from Zenodo record 14182446.

This script shows:
1. How to download data from Zenodo
2. How to load EMG signals and fatigue labels
3. How to preprocess the data
4. How to train models on the Zenodo dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import ZenodoDataLoader, download_zenodo_dataset
from src.preprocessing import EMGPreprocessor
from src.feature_extraction import EMGFeatureExtractor
from src.pipeline import FatigueDetectionPipeline
from sklearn.model_selection import train_test_split
import numpy as np


def download_and_setup():
    """
    Download the Zenodo dataset and set up the data directory.
    """
    print("Step 1: Downloading Zenodo Dataset")
    print("=" * 60)
    
    # Download the dataset
    # Note: If automatic download fails, you'll need to download manually
    try:
        data_dir = download_zenodo_dataset(
            record_id="14182446",
            output_dir="data/zenodo"
        )
        print(f"\nData directory: {data_dir}")
        return data_dir
    except Exception as e:
        print(f"\nAutomatic download not available: {e}")
        print("\nPlease download manually:")
        print("1. Visit: https://zenodo.org/records/14182446")
        print("2. Download all files")
        print("3. Place them in: data/zenodo/")
        return "data/zenodo"


def load_zenodo_data(data_dir="data/zenodo"):
    """
    Load EMG data from the Zenodo dataset.
    
    Args:
        data_dir (str): Directory containing Zenodo data files
        
    Returns:
        tuple: (signals_list, labels_list)
    """
    print("\nStep 2: Loading EMG Data")
    print("=" * 60)
    
    loader = ZenodoDataLoader()
    
    # Load all CSV files from the data directory
    dataset = loader.load_dataset(data_dir)
    
    if not dataset:
        print("No data loaded. Please ensure data files are in the correct directory.")
        print(f"Expected location: {data_dir}")
        return [], []
    
    # Combine all signals and labels
    all_signals = []
    all_labels = []
    
    for filename, (signals, labels) in dataset.items():
        print(f"\nLoaded {filename}:")
        print(f"  - Signals shape: {signals.shape}")
        if labels is not None:
            print(f"  - Labels shape: {labels.shape}")
            print(f"  - Unique labels: {np.unique(labels)}")
        
        all_signals.append(signals)
        if labels is not None:
            all_labels.append(labels)
    
    return all_signals, all_labels


def preprocess_and_train(signals_list, labels_list, sampling_rate=1000):
    """
    Preprocess signals and train fatigue detection models.
    
    Args:
        signals_list (list): List of EMG signal arrays
        labels_list (list): List of label arrays
        sampling_rate (int): Sampling rate of the EMG signals
    """
    print("\nStep 3: Preprocessing and Feature Extraction")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FatigueDetectionPipeline(
        sampling_rate=sampling_rate,
        model_type='svm'  # Start with SVM
    )
    
    # Prepare dataset
    print("\nPreparing dataset...")
    features_df, labels_array = pipeline.prepare_dataset(signals_list, labels_list)
    
    print(f"\nDataset prepared:")
    print(f"  - Features shape: {features_df.shape}")
    print(f"  - Labels shape: {labels_array.shape}")
    print(f"  - Feature columns: {list(features_df.columns)}")
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_array, 
        test_size=0.2, 
        random_state=42,
        stratify=labels_array if len(np.unique(labels_array)) > 1 else None
    )
    
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nStep 4: Training Model")
    print("=" * 60)
    
    pipeline.train(X_train, y_train)
    print("Model trained successfully!")
    
    # Evaluate
    print("\nStep 5: Evaluating Model")
    print("=" * 60)
    
    metrics = pipeline.evaluate(X_test, y_test)
    
    print("\nModel Performance:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    model_path = "models/zenodo_fatigue_model.pkl"
    os.makedirs("models", exist_ok=True)
    pipeline.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return pipeline, metrics


def compare_models(signals_list, labels_list, sampling_rate=1000):
    """
    Compare different models on the Zenodo dataset.
    
    Args:
        signals_list (list): List of EMG signal arrays
        labels_list (list): List of label arrays
        sampling_rate (int): Sampling rate of the EMG signals
    """
    print("\nStep 6: Comparing Multiple Models")
    print("=" * 60)
    
    from src.pipeline import train_multiple_models
    
    # Prepare data
    pipeline = FatigueDetectionPipeline(sampling_rate=sampling_rate)
    features_df, labels_array = pipeline.prepare_dataset(signals_list, labels_list)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels_array,
        test_size=0.2,
        random_state=42,
        stratify=labels_array if len(np.unique(labels_array)) > 1 else None
    )
    
    # Train and compare all models
    results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    print("\nModel Comparison:")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"  Precision: {result['metrics']['precision']:.4f}")
        print(f"  Recall:    {result['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {result['metrics']['f1']:.4f}")
    
    return results


def main():
    """
    Main function to run the complete workflow.
    """
    print("=" * 60)
    print("Zenodo EMG Dataset Example - Muscle Fatigue Detection")
    print("=" * 60)
    
    # Step 1: Download data (or provide instructions for manual download)
    data_dir = download_and_setup()
    
    # Check if data exists
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("\n" + "=" * 60)
        print("DATA NOT FOUND")
        print("=" * 60)
        print("\nPlease download the data manually:")
        print("1. Visit: https://zenodo.org/records/14182446")
        print("2. Download all files")
        print(f"3. Place them in: {data_dir}/")
        print("\nThen run this script again.")
        return
    
    # Step 2: Load data
    signals_list, labels_list = load_zenodo_data(data_dir)
    
    if not signals_list:
        print("\nNo data could be loaded. Please check the data directory and format.")
        return
    
    # Determine sampling rate (you may need to adjust this based on the actual dataset)
    sampling_rate = 1000  # Common EMG sampling rate, adjust if needed
    
    # Step 3-5: Preprocess and train a single model
    pipeline, metrics = preprocess_and_train(signals_list, labels_list, sampling_rate)
    
    # Step 6: Compare multiple models (optional)
    print("\n" + "=" * 60)
    user_input = input("\nWould you like to compare all models (KNN, SVM, Logistic Regression)? (y/n): ")
    if user_input.lower() == 'y':
        compare_models(signals_list, labels_list, sampling_rate)
    
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
    print("\nYou can now:")
    print("- Use the trained model for predictions")
    print("- Analyze the results further")
    print("- Try different hyperparameters")
    print("- Explore the notebooks for more examples")


if __name__ == "__main__":
    main()
