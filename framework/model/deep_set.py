from typing import Literal, Optional, Tuple

import torch
from torch import nn


class DeepSetRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        phi_hidden_dims: Tuple[int] = (16, 4),
        rho_hidden_dims: Tuple[int] = (16, 4),
        dropout: float = 0.1,
        pooling: Literal["sum", "mean"] = "mean",
    ) -> None:
        super().__init__()

        self.pooling = pooling.lower()
        assert self.pooling in ["sum", "mean"], (
            f"Expected pooling to be 'sum' or 'mean', got '{pooling}' instead."
        )

        assert all(dim > 0 for dim in phi_hidden_dims), (
            "All phi hidden dimensions must be positive integers."
        )
        assert all(dim > 0 for dim in rho_hidden_dims), (
            "All rho hidden dimensions must be positive integers."
        )

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.dropout: float = dropout

        phi_dims = [input_dim] + list(phi_hidden_dims)
        self.phi: nn.Module = nn.Sequential(
            *[
                layer
                for i, (din, dout) in enumerate(zip(phi_dims[:-1], phi_dims[1:]))
                for layer in (
                    [nn.Linear(din, dout)]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(phi_dims) - 2
                        else []
                    )
                )
            ]
        )

        rho_dims = [phi_dims[-1]] + list(rho_hidden_dims) + [output_dim]
        self.rho: nn.Module = nn.Sequential(
            *[
                layer
                for i, (din, dout) in enumerate(zip(rho_dims[:-1], rho_dims[1:]))
                for layer in (
                    [nn.Linear(din, dout)]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(rho_dims) - 2
                        else []
                    )
                )
            ]
        )

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # inputs: (batch_size, set_size, input_dim)
        n_batch, n_set, n_x_dim = inputs.shape
        assert n_x_dim == self.input_dim, (
            f"Expected input dim {self.input_dim}, got {n_x_dim}"
        )

        # mask: (batch_size, set_size)
        mask = (
            mask.bool()
            if mask is not None
            else torch.ones((n_batch, n_set), device=inputs.device, dtype=torch.bool)
        )
        assert mask.shape == (n_batch, n_set), (
            f"Expected mask shape {(n_batch, n_set)}, got {mask.shape}"
        )
        valid_mask = mask.unsqueeze(-1).float()

        # (batch_size, set_size, input_dim) -> (batch_size, set_size, phi_out_dim)
        encoded = self.phi(inputs)
        # mask out invalid elements
        masked_encoded = encoded * valid_mask

        # pooling over set elements
        # (batch_size, set_size, phi_out_dim) -> (batch_size, phi_out_dim)
        summed = masked_encoded.sum(dim=1)

        match self.pooling:
            case "sum":
                pooled = summed
            case "mean":
                count = valid_mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / count

        # (batch_size, phi_out_dim) -> (batch_size, output_dim)
        return self.rho(pooled)
