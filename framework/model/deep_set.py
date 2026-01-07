from typing import Literal

import torch
from torch import nn


class DeepSetRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        phi_hidden_dims: int | list[int] | tuple[int, ...] = (16, 4),
        rho_hidden_dims: int | list[int] | tuple[int, ...] = (16, 4),
        dropout: float = 0.1,
        pooling: Literal["sum", "mean"] = "mean",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # validate pooling method
        self.pooling = pooling.lower()
        if self.pooling not in ("sum", "mean"):
            raise ValueError(f"Pooling must be 'sum' or 'mean', got {pooling}")

        # validate hidden dims
        phi_h = (
            [phi_hidden_dims]
            if isinstance(phi_hidden_dims, int)
            else list(phi_hidden_dims)
        )
        rho_h = (
            [rho_hidden_dims]
            if isinstance(rho_hidden_dims, int)
            else list(rho_hidden_dims)
        )

        # element-wise encoder phi
        # (..., input_dim) -> (..., phi_out_dim)
        phi_dims = [input_dim] + phi_h
        self.phi = nn.Sequential(
            *[
                layer
                for i, (d_in, d_out) in enumerate(zip(phi_dims[:-1], phi_dims[1:]))
                for layer in (
                    [nn.Linear(d_in, d_out)]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(phi_dims) - 2
                        else []
                    )
                )
            ]
        )

        # global pooling decoder rho
        # (..., phi_out_dim) -> (..., output_dim)
        rho_dims = [phi_dims[-1]] + rho_h + [output_dim]
        self.rho = nn.Sequential(
            *[
                layer
                for i, (d_in, d_out) in enumerate(zip(rho_dims[:-1], rho_dims[1:]))
                for layer in (
                    [nn.Linear(d_in, d_out)]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(rho_dims) - 2
                        else []
                    )
                )
            ]
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # inputs: (batch, set_size, input_dim)
        # mask: (batch, set_size) - True for valid positions, False for padding
        batch_size, set_size, feat_dim = inputs.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {feat_dim}")

        # handle mask: (batch, set_size) -> (batch, set_size, 1)
        if mask is None:
            mask_val = torch.ones((batch_size, set_size, 1), device=inputs.device)
        else:
            mask = mask.bool()
            mask_val = mask.unsqueeze(-1).to(inputs.dtype)

        # element-wise transformation
        # (batch, set_size, input_dim) -> (batch, set_size, phi_out_dim)
        encoded = self.phi(inputs) * mask_val

        # permutation-invariant pooling
        # (batch, set_size, phi_out_dim) -> (batch, phi_out_dim)
        summed = encoded.sum(dim=1)
        if self.pooling == "sum":
            pooled = summed
        else:
            counts = mask_val.sum(dim=1)
            pooled = summed / counts.clamp(min=1.0)

        # (batch_size, phi_out_dim) -> (batch_size, output_dim)
        return self.rho(pooled)
