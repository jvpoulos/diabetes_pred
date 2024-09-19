import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchmetrics.classification import BinaryAUROC
import os
from pathlib import Path
import argparse
import glob

def load_predictions_and_labels(predictions_dir: Path, labels_dir: Path) -> dict:
    data = {}
    for split in ['val', 'test']:
        pred_pattern = f"{split}_predictions_epoch_*.pt"
        label_pattern = f"{split}_labels_epoch_*.pt"
        
        pred_files = glob.glob(str(predictions_dir / pred_pattern))
        label_files = glob.glob(str(labels_dir / label_pattern))
        
        if pred_files and label_files:
            latest_pred_file = max(pred_files, key=os.path.getmtime)
            latest_label_file = max(label_files, key=os.path.getmtime)
            
            epoch = int(latest_pred_file.split('_')[-1].split('.')[0])
            
            predictions = torch.load(latest_pred_file, map_location=torch.device('cpu'))
            labels = torch.load(latest_label_file, map_location=torch.device('cpu'))
            
            print(f"\nLoaded {split} data from epoch {epoch}")
            print(f"Predictions file: {latest_pred_file}")
            print(f"Labels file: {latest_label_file}")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Predictions dtype: {predictions.dtype}")
            print(f"Labels dtype: {labels.dtype}")
            print(f"Logits range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
            print(f"First 10 logits: {predictions[:10].tolist()}")
            
            # Convert logits to probabilities using PyTorch's sigmoid
            probabilities = predictions
            print(f"Probabilities range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")
            print(f"First 10 probabilities: {probabilities[:10].tolist()}")
            print(f"First 10 labels: {labels[:10].tolist()}")
            print(f"Label distribution: {torch.unique(labels, return_counts=True)}")
            
            # Calculate BinaryAUROC
            auroc = BinaryAUROC()
            auc_score = auroc(probabilities, labels.int())
            print(f"BinaryAUROC: {auc_score:.4f}")
            
            # Check for mismatches
            mismatch_count = ((probabilities > 0.5).float() != labels).sum().item()
            print(f"Number of mismatches: {mismatch_count}")
            print(f"Accuracy: {1 - mismatch_count / len(labels):.4f}")
            
            data[split] = {
                'logits': predictions.numpy(),
                'probabilities': probabilities.numpy(),
                'labels': labels.numpy()
            }
        else:
            print(f"No files found for {split}")
    
    return data

def plot_confusion_matrix(y_true, y_pred, save_path, split):
    cm = confusion_matrix(y_true, y_pred > 0.5)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({split.capitalize()} Set)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def plot_auc_curve(y_true, y_pred, save_path, split):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({split.capitalize()} Set)')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize model results.")
    parser.add_argument("--predictions_dir", type=Path, required=True, help="Path to the predictions directory")
    parser.add_argument("--labels_dir", type=Path, required=True, help="Path to the labels directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to save the output visualizations")
    args = parser.parse_args()

    data = load_predictions_and_labels(args.predictions_dir, args.labels_dir)

    if not data:
        print("Error: No data could be loaded. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['val', 'test']:
        if split in data:
            y_true = data[split]['labels']
            y_pred = data[split]['probabilities']

            print(f"\nCreating visualizations for {split} set")
            print(f"Labels shape: {y_true.shape}")
            print(f"Predictions shape: {y_pred.shape}")
            print(f"Labels range: [{y_true.min():.4f}, {y_true.max():.4f}]")
            print(f"Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

            plot_confusion_matrix(y_true, y_pred, args.output_dir / f'{split}_confusion_matrix.png', split)
            plot_auc_curve(y_true, y_pred, args.output_dir / f'{split}_auc_curve.png', split)
            print(f"Visualizations for {split} set saved")
        else:
            print(f"Skipping {split} set: missing data")

    print("\nVisualizations completed. Results saved in the output directory.")

if __name__ == "__main__":
    main()