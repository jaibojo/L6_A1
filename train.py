import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import MNISTModel
from logger import TrainingLogger

class CustomScheduler:
    def __init__(self, optimizer, epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.current_step = 0
        
        # Adjusted LR ranges
        self.lr_ranges = {
            0: (0.005, 0.02),     # Epoch 1: Keep the good warmup
            1: (0.025, 0.035),    # Epoch 2: Higher LR range
            2: (0.02, 0.001)      # Epoch 3: Start higher, then cooldown
        }
    
    def step(self):
        epoch = self.current_step // self.steps_per_epoch
        step_in_epoch = self.current_step % self.steps_per_epoch
        
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            
        min_lr, max_lr = self.lr_ranges[epoch]
        
        if epoch == 0:
            # Epoch 1: Keep successful linear warmup
            progress = step_in_epoch / self.steps_per_epoch
            lr = min_lr + (max_lr - min_lr) * progress
        elif epoch == 1:
            # Epoch 2: Step-wise increase with holds
            progress = step_in_epoch / self.steps_per_epoch
            if progress < 0.3:
                lr = min_lr
            elif progress < 0.6:
                lr = (min_lr + max_lr) / 2
            else:
                lr = max_lr
        else:
            # Epoch 3: Cosine decay
            progress = step_in_epoch / self.steps_per_epoch
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_step += 1
        
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 32
    epochs = 3
    
    # Data loading
    train_transform = transforms.Compose([
        transforms.RandomRotation(8),
        transforms.RandomAffine(0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    train_dataset, _ = random_split(full_dataset, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, optimizer
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.01,  # Starting LR
        weight_decay=0.003, 
        betas=(0.95, 0.999)
    )
    scheduler = CustomScheduler(optimizer, epochs, len(train_loader))
    
    # Initialize logger
    logger = TrainingLogger()
    logger.log_model_summary(model)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        min_lr = float('inf')
        max_lr = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            current_lr = scheduler.get_last_lr()[0]
            min_lr = min(min_lr, current_lr)
            max_lr = max(max_lr, current_lr)
            
            # Log progress
            accuracy = 100. * correct / total
            logger.log_batch_progress(epoch + 1, batch_idx, len(train_loader),
                                    loss.item(), current_lr, accuracy)
            
            scheduler.step()
        
        # Calculate final training accuracy for epoch
        train_accuracy = 100. * correct / total
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        avg_loss = running_loss / len(train_loader)
        
        # Update epoch stats with both train and test accuracy
        logger.update_epoch_stats(epoch + 1, train_accuracy, test_accuracy, min_lr, max_lr, avg_loss)
        print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    logger.print_summary_table()

if __name__ == "__main__":
    train() 