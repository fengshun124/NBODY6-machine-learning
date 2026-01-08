import torch
from torch import nn


class SummaryStatsRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: int | list[int] | tuple[int, ...] = (16, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # count (1) + per-feature statistics (7 * input_dim):
        # mean, median, min, max, std, 25th percentile, 75th percentile
        self.feat_dims = input_dim * 7 + 1

        # normalize hidden dims to list
        h_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)

        # multi-layer perceptron
        layer_dims = [self.feat_dims] + h_dims + [output_dim]
        self.mlp = nn.Sequential(
            *[
                layer
                for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:]))
                for layer in (
                    [nn.Linear(d_in, d_out)]
                    + (
                        [nn.ReLU(), nn.Dropout(dropout)]
                        if i < len(layer_dims) - 2
                        else []
                    )
                )
            ]
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # inputs: (batch, set_size, input_dim)
        batch_size, set_size, feat_dim = inputs.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {feat_dim}")

        # handle mask: (batch, set_size) -> (batch, set_size, 1)
        if mask is None:
            mask_val = torch.ones((batch_size, set_size, 1), device=inputs.device)
        else:
            mask = mask.bool()
            mask_val = mask.unsqueeze(-1).to(inputs.dtype)

        count = mask_val.sum(dim=1)
        safe_count = count.clamp(min=1.0)
        is_empty = count == 0

        # mean and std.
        masked_inputs_zero = inputs.masked_fill(mask_val == 0, 0.0)
        mean_val = masked_inputs_zero.sum(dim=1) / safe_count

        diffs = (inputs - mean_val.unsqueeze(1)).masked_fill(mask_val == 0, 0.0)
        var_val = diffs.pow(2).sum(dim=1) / (safe_count - 1).clamp(min=1.0)
        std_val = var_val.sqrt()

        # percentiles: 25th, 50th (median), 75th
        masked_inputs_nan = inputs.masked_fill(mask_val == 0, float("nan"))
        q_vals = torch.nanquantile(
            masked_inputs_nan,
            q=torch.tensor([0.25, 0.5, 0.75], device=inputs.device),
            dim=1,
        )

        # min and max
        min_val = inputs.masked_fill(mask_val == 0, float("inf")).min(dim=1)[0]
        max_val = inputs.masked_fill(mask_val == 0, float("-inf")).max(dim=1)[0]

        stats = [
            count,
            mean_val,
            # median
            q_vals[1],
            min_val,
            max_val,
            std_val,
            # 25th percentile
            q_vals[0],
            # 75th percentile
            q_vals[2],
        ]
        pooled_stats = torch.cat(
            [torch.where(is_empty, torch.zeros_like(s), s) for s in stats], dim=1
        )
        # (batch_size, 1 + 7 * input_dim [feat_dims]) -> (batch_size, output_dim)
        return self.mlp(pooled_stats)
