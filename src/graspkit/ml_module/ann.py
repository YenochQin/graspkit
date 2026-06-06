"""Standard feed-forward ANN backbone for CSF descriptors."""

from __future__ import annotations

import torch
import torch.nn as nn


class StandardANN(nn.Module):
    """Fully connected ANN for flat CSF descriptor vectors."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 150,
        output_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the standard feed-forward ANN.

        Args:
            input_size: Number of flat descriptor features.
            hidden_size: Width of the first hidden layer.
            output_size: Number of output logits.
            dropout: Dropout probability between dense layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear-layer weights and biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from flat CSF descriptors."""
        return self.layers(x)
