import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the Basic_Block, a 1x1 convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(Basic_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        """
        Applies a 1x1 convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height).

        Returns:
            torch.Tensor: Output tensor after applying 1x1 convolution.
        """
        return self.conv(x)


class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        """
        Initializes the Residual_Block, which applies two 3x3 convolution layers.

        Args:
            num_channels (int): Number of input channels (and output channels).
        """
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height).

        Returns:
            torch.Tensor: Output tensor after two convolutions and ReLU activations.
        """
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out) + x, inplace=True)  # Residual connection
        return out


class Cascading_Block(nn.Module):
    def __init__(self, channels):
        """
        Initializes the Cascading_Block, which contains multiple residual blocks
        and concatenates their outputs with the input, followed by 1x1 convolution.

        Args:
            channels (int): Number of channels for the convolutions.
        """
        super(Cascading_Block, self).__init__()
        self.r1 = Residual_Block(channels)
        self.r2 = Residual_Block(channels)
        self.r3 = Residual_Block(channels)
        self.c1 = Basic_Block(channels * 2, channels)
        self.c2 = Basic_Block(channels * 3, channels)
        self.c3 = Basic_Block(channels * 4, channels)

    def forward(self, x):
        """
        Forward pass for the cascading block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height).

        Returns:
            torch.Tensor: Output tensor after cascading residual blocks and 1x1 convolutions.
        """
        # First residual block
        c0 = o0 = x
        b1 = self.r1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        # Second residual block
        b2 = self.r2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        # Third residual block
        b3 = self.r3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
