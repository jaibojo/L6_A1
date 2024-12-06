import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, dropout_rate1=0.05, dropout_rate2=0.1):
        super(MNISTModel, self).__init__()
        
        # First conv block
        self.block1 = nn.Sequential(
            # params = (5*5*1*4) + 4 = 104
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),  # 28x28*1 -> 28x28*4
            nn.MaxPool2d(2, 2)  # 28x28*4 -> 14x14*4 (no params)
        )
        
        # First activation block
        self.act1 = nn.Sequential(
            # params = 4*2 (running mean/var) + 4*2 (weight/bias) = 16
            nn.BatchNorm2d(4),
            nn.ReLU(),  # no params
            nn.Dropout2d(dropout_rate1)  # no params
        )
        
        # Second conv block
        self.block2 = nn.Sequential(
            # params = (3*3*4*8) + 8 = 296
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),  # 14x14*4 -> 14x14*8
            nn.MaxPool2d(2, 2)  # 14x14*8 -> 7x7*8 (no params)
        )
        
        # Second activation block
        self.act2 = nn.Sequential(
            # params = 8*2 (running mean/var) + 8*2 (weight/bias) = 32
            nn.BatchNorm2d(8),
            nn.ReLU(),  # no params
            nn.Dropout2d(dropout_rate2)  # no params
        )
        
        # Final conv block with stride=2 to get to 4x4
        self.block3 = nn.Sequential(
            # params = (3*3*8*8) + 8 = 584
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # 7x7*8 -> 4x4*8
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 4*4*8 = 128 features (no params)
            # params = (128*12) + 12 = 1,548
            nn.Linear(8 * 4 * 4, 12),
            nn.ReLU(),  # no params
            # params = (12*10) + 10 = 130
            nn.Linear(12, 10)
        )
        
    def forward(self, x):
        # Conv blocks with activations
        x = self.block1(x)
        x = self.act1(x)
        x = self.block2(x)
        x = self.act2(x)
        x = self.block3(x)
        
        # FC layers
        x = self.fc_layers(x)
        return x

# Total params = 104 + 16 + 296 + 32 + 584 + 1,548 + 130 = 2,710 parameters