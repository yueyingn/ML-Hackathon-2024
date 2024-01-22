import numpy as np
import torch
from torch import einsum, nn, pi, softmax

@torch.no_grad()
def zero_init(module: torch.nn.Module) -> torch.nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module

# the residual block class
class ResnetBlock(nn.Module):
    def __init__(self, ch_in, ch_out=None, condition_dim=None, dropout_prob=0.0, norm_groups=32):
        super().__init__()
        # Set output channels to input channels if not specified
        ch_out = ch_in if ch_out is None else ch_out

        # Store the channel and condition dimensions
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.condition_dim = condition_dim

        # First part of the network
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, padding_mode="circular"),
        )

        # Conditional projection if condition_dim is specified
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))

        # Second part of the network
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, padding_mode="circular")),
        )

        # Skip connection if input and output channels differ
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        # Apply the first part of the network
        h = self.net1(x)

        # Apply conditional projection if condition is provided
        if condition is not None:
            # Ensure condition shape matches the batch size
            assert condition.shape == (x.shape[0], self.condition_dim)
            condition = self.cond_proj(condition)
            # Expand condition to match the spatial dimensions of x
            condition = condition[:, :, None, None]  # 2D
            # Add the condition to the output of the first network part
            h = h + condition

        # Apply the second part of the network
        h = self.net2(h)

        # Apply skip connection if needed
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)

        # Ensure the shapes of the input and the processed input match
        assert x.shape == h.shape
        # Return the sum of the input and the processed input
        return x + h
