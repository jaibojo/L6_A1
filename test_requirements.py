import torch
import torch.nn as nn
from model import MNISTModel
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def test_data_split():
    """Test if data is split 50k/10k"""
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
    
    assert len(train_dataset) == 50000, f"Training set should be 50k, got {len(train_dataset)}"
    assert len(val_dataset) == 10000, f"Validation set should be 10k, got {len(val_dataset)}"
    print("✓ Data split is correct (50k/10k)")

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_params():
    """Test if model has less than 20k parameters"""
    model = MNISTModel()
    num_params = count_parameters(model)
    
    assert num_params < 20000, f"Model should have <20k params, got {num_params}"
    print(f"✓ Model has {num_params} parameters (<20k)")

def test_model_architecture():
    """Test if model uses BatchNorm and Dropout"""
    model = MNISTModel()
    
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    
    assert has_batchnorm, "Model should use BatchNorm"
    assert has_dropout, "Model should use Dropout"
    print("✓ Model uses BatchNorm and Dropout")

def test_training_epochs():
    """Test if training uses less than 20 epochs"""
    with open('train.py', 'r') as f:
        content = f.read()
        assert 'epochs = ' in content, "Couldn't find epochs definition"
        
        # Extract epochs value more robustly
        for line in content.split('\n'):
            if 'epochs = ' in line and not line.strip().startswith('#'):
                try:
                    # Handle potential spaces and comments
                    value = line.split('=')[1].strip().split('#')[0].strip()
                    epochs = int(value)
                    assert epochs < 20, f"Should use <20 epochs, got {epochs}"
                    print(f"✓ Training uses {epochs} epochs (<20)")
                    return
                except (ValueError, IndexError):
                    continue
        
        raise AssertionError("Could not find valid epochs value in train.py")

def test_accuracy_requirement():
    """Test if the model achieves the required accuracy"""
    try:
        with open("accuracy.log", "r") as log_file:
            lines = log_file.readlines()
            last_line = lines[-1]
            accuracy_str = last_line.split("Test Accuracy: ")[1].strip().replace("%", "")
            accuracy = float(accuracy_str)
            assert accuracy >= 99.4, f"Test accuracy should be >= 99.4%, got {accuracy}%"
            print(f"✓ Test accuracy requirement met: {accuracy}%")
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise AssertionError("Could not verify test accuracy from logs")

def main():
    print("\nTesting Project Requirements:")
    print("=============================")
    
    try:
        test_data_split()
        test_model_params()
        test_model_architecture()
        test_training_epochs()
        test_accuracy_requirement()
        
        print("\nAll basic requirements met! ✓")
        
    except AssertionError as e:
        print(f"\n❌ Test Failed: {str(e)}")

if __name__ == "__main__":
    main() 