import pytorch_lightning as pl
import torch
import torch.nn as nn


class SelfAttnBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int) -> None:
        super(SelfAttnBlock, self).__init__()
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query

        q_proj = self.query_proj(query)
        k_proj = self.key_proj(key)
        v_proj = self.value_proj(value)

        attn_output, _ = self.attention(q_proj, k_proj, v_proj)
        norm_output = self.norm1(q_proj + attn_output)
        ff_output = self.feed_forward(norm_output)
        final_output = self.norm2(norm_output + ff_output)
        return final_output


class InducedSetAttnBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_heads: int, num_inducing: int
    ) -> None:
        super(InducedSetAttnBlock, self).__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)

        self.induced_point = nn.Parameter(torch.randn(num_inducing, output_dim))

        self.norm = nn.LayerNorm(output_dim)
        self.encoder = SelfAttnBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            num_heads=num_heads,
        )
        self.decoder = SelfAttnBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        repeated_tokens = self.induced_point.unsqueeze(0).expand(x_proj.size(0), -1, -1)

        encoded_x = self.encoder(x_proj)
        induced_out_intermediate = self.decoder(repeated_tokens, encoded_x, encoded_x)

        induced_out_pooled = induced_out_intermediate.mean(dim=1, keepdim=True)
        induced_out_expanded = induced_out_pooled.expand(-1, x_proj.size(1), -1)

        output = x_proj + induced_out_expanded
        return self.norm(output)


class PoolingMultiHeadAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int) -> None:
        super(PoolingMultiHeadAttn, self).__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.multi_head_attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, encoded_set: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_set.size(0)
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        pooled_output, _ = self.multi_head_attention(seeds, encoded_set, encoded_set)
        return self.norm(pooled_output)


class NBodyTransformerRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        num_inducing: int,
        num_seeds: int,
        num_sab: int = 1,
        lr: float = 1e-3,
    ) -> None:
        super(NBodyTransformerRegressor, self).__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            InducedSetAttnBlock(
                input_dim=input_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
                num_inducing=num_inducing,
            ),
            *[
                SelfAttnBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_sab - 1)
            ],
        )

        self.pooling = PoolingMultiHeadAttn(
            dim=hidden_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
        )
        self.regressor = nn.Linear(hidden_dim * num_seeds, output_dim)

        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_set = self.encoder(x)
        pooled_output = self.pooling(encoded_set)
        flatten_output = pooled_output.view(pooled_output.size(0), -1)
        return self.regressor(flatten_output)

    def _evaluation(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        mse_loss = self.mse_loss_fn(y_hat, y)
        self.log(f"{stage}_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        mae_loss = self.mae_loss_fn(y_hat, y)
        self.log(f"{stage}_mae", mae_loss, on_step=False, on_epoch=True, prog_bar=True)

        for i in range(y_hat.size(1)):
            self.log(
                f"{stage}_attr{i}_mse",
                self.mse_loss_fn(y_hat[:, i], y[:, i]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{stage}_attr{i}_mae",
                self.mae_loss_fn(y_hat[:, i], y[:, i]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        return mse_loss

    def training_step(self, batch, batch_idx):
        return self._evaluation(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._evaluation(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
