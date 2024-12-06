import sys
from typing import Dict, List
from tabulate import tabulate
import torch.nn as nn

class TrainingLogger:
    def __init__(self):
        self.epoch_stats = []
        self.current_epoch_data = {}
        
    def log_model_summary(self, model: nn.Module):
        """Print model architecture and parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        print("\n" + "="*50)
        print("Model Architecture:")
        print(model)
        print(f"\nTotal Parameters: {total_params:,}")
        print("="*50 + "\n")
        
    def log_batch_progress(self, epoch: int, batch: int, total_batches: int, 
                          loss: float, lr: float, accuracy: float):
        """Log progress during training"""
        sys.stdout.write('\r')
        sys.stdout.write(
            f'Epoch: {epoch} | Batch: {batch}/{total_batches} | '
            f'Loss: {loss:.4f} | LR: {lr:.6f} | Acc: {accuracy:.2f}%'
        )
        sys.stdout.flush()
        
    def update_epoch_stats(self, epoch: int, train_accuracy: float, test_accuracy: float, 
                          min_lr: float, max_lr: float, avg_loss: float):
        """Store statistics for each epoch"""
        self.epoch_stats.append({
            'Epoch': epoch,
            'Train Acc': f"{train_accuracy:.2f}%",
            'Test Acc': f"{test_accuracy:.2f}%",
            'Max LR': f"{max_lr:.6f}",
            'Min LR': f"{min_lr:.6f}",
            'Avg Loss': f"{avg_loss:.4f}"
        })
        
    def print_summary_table(self):
        """Print final summary table"""
        print("\n\nTraining Summary:")
        print(tabulate(self.epoch_stats, headers="keys", tablefmt="grid")) 