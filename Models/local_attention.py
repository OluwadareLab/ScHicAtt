import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, in_dim, window_size=7):
        """
        Initializes the LocalAttention module.

        Args:
            in_dim (int): The number of input channels.
            window_size (int): The size of the local window to focus attention on (default 7).
        """
        super(LocalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.window_size = window_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for local attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height)

        Returns:
            torch.Tensor: Output tensor with local attention applied.
        """
        m_batchsize, C, width, height = x.size()

        # Create query, key, and value matrices for local attention
        proj_query = self._local_window(self.query_conv(x), self.window_size)  # B x N x C
        proj_key = self._local_window(self.key_conv(x), self.window_size)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N (energy map for local window)
        attention = self.softmax(energy)  # Apply softmax to normalize attention scores
        proj_value = self._local_window(self.value_conv(x), self.window_size)  # B x C x N

        # Multiply attention map with the value matrix
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N

        # Reshape and recombine the local windows into the original image shape
        out = self._recombine_windows(out, m_batchsize, C, width, height, self.window_size)
        
        # Apply learnable scaling factor gamma
        out = self.gamma * out + x
        return out

    def _local_window(self, x, window_size):
        """
        Breaks input tensor into smaller local windows.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, width, height)
            window_size (int): The size of the window to split the tensor into.

        Returns:
            torch.Tensor: Reshaped tensor into local windows.
        """
        B, C, W, H = x.shape
        x_unfold = F.unfold(x, kernel_size=window_size, stride=window_size)  # B x (C*window_size*window_size) x num_windows
        x_unfold = x_unfold.view(B, C, window_size * window_size, -1)  # B x C x window_size^2 x num_windows
        return x_unfold.permute(0, 3, 2, 1).contiguous()  # B x num_windows x window_size^2 x C

    def _recombine_windows(self, out, batch_size, channels, width, height, window_size):
        """
        Recombines local windows into the original image shape.

        Args:
            out (torch.Tensor): Tensor of processed local windows.
            batch_size (int): Number of samples in the batch.
            channels (int): Number of channels in the input.
            width (int): Width of the input image.
            height (int): Height of the input image.
            window_size (int): Size of the window used.

        Returns:
            torch.Tensor: Output tensor with windows recombined.
        """
        num_windows = (width // window_size) * (height // window_size)
        out = out.permute(0, 3, 1, 2).contiguous().view(batch_size, channels * window_size * window_size, num_windows)
        out = F.fold(out, output_size=(width, height), kernel_size=window_size, stride=window_size)
        return out
