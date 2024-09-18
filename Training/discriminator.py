import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=64):
        """
        Initialize the Discriminator network.
        
        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale Hi-C matrices).
            num_features (int): Number of features in the first convolutional layer.
        """
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_features * 4, num_features * 8, kernel_size=4, stride=2, padding=1)
        
        # Fully connected layer to output a single value (real or fake classification)
        self.fc = nn.Linear(num_features * 8 * 16 * 16, 1)  # Adjust the input size based on your Hi-C data size

    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor (Hi-C matrix) of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Discriminator score (real or fake).
        """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        
        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)  # Flatten the tensor to feed it into the fully connected layer
        
        # Output a single value (real or fake score)
        out = torch.sigmoid(self.fc(x))
        return out
