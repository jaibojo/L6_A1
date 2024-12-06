# MNIST Classification Project

A PyTorch implementation of a lightweight CNN model for MNIST digit classification with specific constraints and requirements.

## Project Requirements

- [x] Model Parameters: < 20,000
- [x] Training Data Split: 50,000 train / 10,000 validation
- [x] Training Duration: < 20 epochs
- [x] Architecture Features: BatchNorm and Dropout
- [x] Target Accuracy: 99.4% on validation/test set

## Model Architecture

The model (`MNISTModel`) is a convolutional neural network with:

- **First Convolutional Block:**
  - Conv2d: 1 input channel, 4 output channels, 3x3 kernel, stride 1, padding 1
  - MaxPool2d: 2x2 kernel, stride 2
  - BatchNorm2d
  - ReLU activation
  - Dropout2d (5%)

- **Second Convolutional Block:**
  - Conv2d: 4 input channels, 8 output channels, 3x3 kernel, stride 1, padding 1
  - MaxPool2d: 2x2 kernel, stride 2
  - BatchNorm2d
  - ReLU activation
  - Dropout2d (10%)

- **Final Convolutional Block:**
  - Conv2d: 8 input channels, 8 output channels, 3x3 kernel, stride 2, padding 1

- **Fully Connected Layers:**
  - Flatten
  - Linear: 128 input features, 12 output features
  - ReLU activation
  - Linear: 12 input features, 10 output features

Total parameters: 2,710

## Training Approach

The training uses a staged learning approach with curriculum learning:

### Stages
1. Stage 1: 10% easiest samples (lr=0.001)
2. Stage 2: 15% next samples (lr=0.0008)
3. Stage 3: 20% medium samples (lr=0.0006)
4. Stage 4: 25% harder samples (lr=0.0004)
5. Stage 5: 30% hardest samples (lr=0.0002)

### Training Details
- Epochs: 20
- Early stopping patience: 5
- Batch size: 32
- Dataset split: 50,000 training, 10,000 testing
- Optimizer: Adam with stage-specific learning rates
- Loss: CrossEntropy with label smoothing (0.1)

## Requirements

- Python 3.11
- PyTorch 2.5.1
- TorchVision 0.20.1
- pytest ≥ 6.0.0
- matplotlib ≥ 3.7.1