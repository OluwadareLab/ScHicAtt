import torch
import torch.nn as nn
from .self_attention import SelfAttention
from .local_attention import LocalAttention
from .global_attention import GlobalAttention
from .dynamic_attention import DynamicAttention
from .cascading_block import Cascading_Block


class Generator(nn.Module):
    def __init__(self, num_channels, attention_type="self"):
        """
        Initializes the Generator model with a specified attention mechanism.

        Args:
            num_channels (int): Number of channels for the convolutional layers.
            attention_type (str): Type of attention to apply ("self", "local", "global", "dynamic").
        """
        super().__init__()

        # Entry 3x3 convolution layer
        self.entry = nn.Conv2d(1, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Cascading blocks
        self.cb1 = Cascading_Block(num_channels)
        self.cb2 = Cascading_Block(num_channels)
        self.cb3 = Cascading_Block(num_channels)
        self.cb4 = Cascading_Block(num_channels)
        self.cb5 = Cascading_Block(num_channels)

        # Select attention mechanism
        if attention_type == "self":
            self.att1 = SelfAttention(num_channels)
            self.att2 = SelfAttention(num_channels)
        elif attention_type == "local":
            self.att1 = LocalAttention(num_channels)
            self.att2 = LocalAttention(num_channels)
        elif attention_type == "global":
            self.att1 = GlobalAttention(num_channels)
            self.att2 = GlobalAttention(num_channels)
        elif attention_type == "dynamic":
            self.att1 = DynamicAttention(num_channels)
            self.att2 = DynamicAttention(num_channels)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Body 1x1 convolution layers
        self.cv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv2 = nn.Conv2d(num_channels * 3, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv3 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv4 = nn.Conv2d(num_channels * 5, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.cv5 = nn.Conv2d(num_channels * 6, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # 3x3 exit convolution layer
        self.exit = nn.Conv2d(num_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """
        Forward pass for the Generator model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.entry(x)
        c0 = o0 = x

        # First cascading block + attention
        b1 = self.cb1(o0)
        a1 = self.att1(b1)  # Apply attention after the first cascading block
        c1 = torch.cat([c0, a1], dim=1)
        o1 = self.cv1(c1)

        # Second cascading block + attention
        b2 = self.cb2(o1)
        a2 = self.att2(b2)  # Apply attention after the second cascading block
        c2 = torch.cat([c1, a2], dim=1)
        o2 = self.cv2(c2)

        # Continue with cascading blocks and convolutions
        b3 = self.cb3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.cv3(c3)

        b4 = self.cb4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.cv4(c4)

        b5 = self.cb5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.cv5(c5)

        out = self.exit(o5)
        return out
