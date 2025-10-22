"""
Visualization module for EMG data and model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_emg_signal(signal_data, sampling_rate=1000, title="EMG Signal"):
    """
    Plot an EMG signal.
    
    Args:
        signal_data (np.ndarray): EMG signal
        sampling_rate (int): Sampling rate in Hz
        title (str): Plot title
    """
    time = np.arange(len(signal_data)) / sampling_rate
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison_signals(signals, labels, sampling_rate=1000):
    """
    Plot multiple EMG signals for comparison.
    
    Args:
        signals (list): List of EMG signals
        labels (list): Labels for each signal
        sampling_rate (int): Sampling rate in Hz
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (signal_data, label) in enumerate(zip(signals, labels)):
        time = np.arange(len(signal_data)) / sampling_rate
        axes[i].plot(time, signal_data)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'EMG Signal - {label}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(features_df, labels, feature_names=None):
    """
    Plot distributions of features for different fatigue levels.
    
    Args:
        features_df (pd.DataFrame): Feature data
        labels (np.ndarray): Fatigue labels
        feature_names (list): List of feature names to plot
    """
    if feature_names is None:
        feature_names = features_df.columns.tolist()[:4]  # Plot first 4 features
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    unique_labels = np.unique(labels)
    
    for i, feature in enumerate(feature_names):
        for label in unique_labels:
            mask = labels == label
            axes[i].hist(features_df.loc[mask, feature], alpha=0.5, bins=30, label=f'Class {label}')
        
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{feature} Distribution')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix
        class_names (list): Names of classes
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(conf_matrix))]
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict):
    """
    Compare accuracy of multiple models.
    
    Args:
        results_dict (dict): Dictionary with model results
    """
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]['metrics']['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """
    Plot feature importance.
    
    Args:
        feature_names (list): Names of features
        importances (np.ndarray): Importance scores
        title (str): Plot title
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_training_progress(train_scores, val_scores, metric='accuracy'):
    """
    Plot training progress over epochs.
    
    Args:
        train_scores (list): Training scores
        val_scores (list): Validation scores
        metric (str): Metric name
    """
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-', label=f'Training {metric}')
    plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training Progress - {metric.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, n_classes=2):
    """
    Plot ROC curve for binary or multi-class classification.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        n_classes (int): Number of classes
    """
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class classification
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
