from typing import Optional, Tuple

import torch
from torch import nn


class SummaryStatsRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int] = (16, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        # count (1) + per-feature statistics (7 * input_dim):
        # mean, median, min, max, std, 25th percentile, 75th percentile
        self.feat_dims = int(self.input_dim * 7 + 1)

        # MLP layers
        layer_dims = [self.feat_dims] + list(hidden_dims) + [output_dim]
        self.mlp = nn.Sequential(
            *[
                layer
                for i in range(len(layer_dims) - 1)
                for layer in (
                    [
                        nn.Linear(
                            in_features=layer_dims[i], out_features=layer_dims[i + 1]
                        )
                    ]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(layer_dims) - 2
                        else []
                    )
                )
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        valid_mask = mask.unsqueeze(-1)

        count = valid_mask.sum(dim=1).float()
        safe_count = count.clamp(min=1.0)
        # check if empty sets
        is_empty = count == 0

        # fill invalid entries with 0 for mean and std computation
        masked_inputs_zero = inputs.masked_fill(~valid_mask, 0.0)
        # mean (batch_size, input_dim)
        mean_val = masked_inputs_zero.sum(dim=1) / safe_count
        # std (batch_size, input_dim)
        diffs = (inputs - mean_val.unsqueeze(1)).masked_fill(~valid_mask, 0.0)
        sq_diffs = diffs.pow(2)
        # safeguard if empty set (count=0)
        var_val = sq_diffs.sum(dim=1) / (safe_count - 1).clamp(min=1.0)
        std_val = var_val.sqrt()

        # fill invalid entries with NaN for percentiles
        masked_inputs_nan = inputs.masked_fill(~valid_mask, float("nan"))
        # 25th, 50th (median), 75th percentiles (3, batch_size, input_dim)
        q_vals = torch.nanquantile(
            masked_inputs_nan,
            q=torch.tensor([0.25, 0.5, 0.75], device=inputs.device),
            dim=1,
        )
        # safeguard if empty sets, set quantiles (median) to 0
        p25_val = torch.where(
            is_empty,
            torch.zeros_like(q_vals[0]),
            q_vals[0],
        )
        median_val = torch.where(
            is_empty,
            torch.zeros_like(q_vals[1]),
            q_vals[1],
        )
        p75_val = torch.where(
            is_empty,
            torch.zeros_like(q_vals[2]),
            q_vals[2],
        )

        # fill invalid entries with +inf/-inf for min/max
        # min (batch_size, input_dim)
        min_inputs = inputs.masked_fill(~valid_mask, float("inf"))
        min_val = min_inputs.min(dim=1)[0]
        min_val = torch.where(
            is_empty,
            torch.zeros_like(min_val),
            min_val,
        )
        # max (batch_size, input_dim)
        max_inputs = inputs.masked_fill(~valid_mask, float("-inf"))
        max_val = max_inputs.max(dim=1)[0]
        max_val = torch.where(
            is_empty,
            torch.zeros_like(max_val),
            max_val,
        )

        # pass pooled statistics through MLP
        pooled_stats = torch.cat(
            [
                count,
                mean_val,
                median_val,
                min_val,
                max_val,
                std_val,
                p25_val,
                p75_val,
            ],
            dim=1,
        )
        # (batch_size, 1 + 7 * input_dim [feat_dims]) -> (batch_size, output_dim)
        return self.mlp(pooled_stats)
