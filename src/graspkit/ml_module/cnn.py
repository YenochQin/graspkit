"""Convolutional backbone for CSF descriptors.

The model follows the 1D CNN block used by Bilous et al. for CSF-level
selection, adapted to GraspKit's PyTorch multi-label BCE-with-logits pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CSFConv1DBackbone(nn.Module):
    """1D convolutional backbone for three-channel CSF descriptors.

    Public GraspKit callers pass flat descriptors with length
    ``3 * n_orbitals``. The module also accepts structured tensors with shape
    ``(batch_size, n_orbitals, 3)`` and converts both forms to Conv1d layout.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        conv1_filters: int = 96,
        conv2_filters: int = 16,
        dense_sizes: tuple[int, int, int] = (150, 120, 90),
    ) -> None:
        """Initialize the CSF Conv1D backbone.

        Args:
            input_size: Flat descriptor length, normally ``3 * n_orbitals``.
            output_size: Number of independent target levels.
            channels: Descriptor channels per orbital.
            conv1_filters: Filters for the kernel-size-3 convolution.
            conv2_filters: Filters for the kernel-size-1 convolution.
            dense_sizes: Dense head sizes.

        Raises:
            ValueError: If ``input_size`` cannot be reshaped into channel
                groups or is too short for the kernel-size-3 convolution.
        """
        super().__init__()
        if input_size % channels != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by channels ({channels})"
            )

        seq_length = input_size // channels
        if seq_length < 3:
            raise ValueError(
                "CSFConv1DBackbone requires at least 3 orbital positions for "
                "Conv1d kernel_size=3"
            )

        self.input_size = input_size
        self.output_size = output_size
        self.channels = channels
        self.seq_length = seq_length
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.dense_sizes = dense_sizes

        self.conv = nn.Sequential(
            nn.Conv1d(channels, conv1_filters, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        flattened_size = conv2_filters * (seq_length - 2)
        dense1, dense2, dense3 = dense_sizes
        self.head = nn.Sequential(
            nn.Linear(flattened_size, dense1),
            nn.ReLU(),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Linear(dense2, dense3),
            nn.ReLU(),
            nn.Linear(dense3, output_size),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize trainable layers with stable defaults."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for flat or structured CSF descriptors."""
        if x.dim() == 2:
            if x.shape[1] != self.input_size:
                raise ValueError(
                    f"Expected flat input with {self.input_size} features, "
                    f"got {x.shape[1]}"
                )
            x = x.reshape(x.shape[0], self.seq_length, self.channels)
        elif x.dim() == 3:
            expected_shape = (self.seq_length, self.channels)
            actual_shape = (x.shape[1], x.shape[2])
            if actual_shape != expected_shape:
                raise ValueError(
                    "Expected structured input shape "
                    f"(*, {expected_shape[0]}, {expected_shape[1]}), "
                    f"got (*, {actual_shape[0]}, {actual_shape[1]})"
                )
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")

        return self.head(self.conv(x.transpose(1, 2)))
