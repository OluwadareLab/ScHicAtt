import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height)

        Returns:
            torch.Tensor: Output tensor after applying self-attention mechanism.
        """
        m_batchsize, C, width, height = x.size()

        # Create query, key, and value matrices
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N (energy map)
        attention = self.softmax(energy)  # Apply softmax to normalize attention scores
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N

        # Multiply attention map with the value matrix
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, width, height)  # Reshape to match original input dimensions

        # Apply learnable scaling factor gamma
        out = self.gamma * out + x
        return out
