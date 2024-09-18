import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):
    def __init__(self, in_dim):
        """
        Initializes the GlobalAttention module.

        Args:
            in_dim (int): The number of input channels.
        """
        super(GlobalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for global attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height)

        Returns:
            torch.Tensor: Output tensor with global attention applied.
        """
        m_batchsize, C, width, height = x.size()

        # Create query, key, and value matrices for global attention
        proj_query = self.query_conv(x).view(m_batchsize, -1)  # B x (C * width * height)
        proj_key = self.key_conv(x).view(m_batchsize, -1).permute(0, 2, 1)  # B x (C * width * height)
        energy = torch.bmm(proj_query.unsqueeze(2), proj_key.unsqueeze(1))  # B x 1 x 1 (global energy map)
        attention = self.softmax(energy)  # Apply softmax to normalize attention scores
        proj_value = self.value_conv(x).view(m_batchsize, -1).unsqueeze(2)  # B x (C * width * height) x 1

        # Multiply attention map with the value matrix (global scale)
        out = torch.bmm(proj_value, attention)  # B x (C * width * height) x 1
        out = out.view(m_batchsize, C, width, height)  # Reshape to match original input dimensions

        # Apply learnable scaling factor gamma
        out = self.gamma * out + x
        return out
