from typing import Optional

import torch
import torch.nn as nn


class MultiheadAttnBlock(nn.Module):
    def __init__(
        self,
        q_in_dim: int,
        k_in_dim: int,
        v_in_dim: int,
        num_heads: int,
        ln: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.v_in_dim = v_in_dim
        self.num_heads = num_heads

        # projections
        self.fc_q = nn.Linear(in_features=q_in_dim, out_features=v_in_dim)
        self.fc_k = nn.Linear(in_features=k_in_dim, out_features=v_in_dim)
        self.fc_v = nn.Linear(in_features=k_in_dim, out_features=v_in_dim)

        # layer norms
        self.ln0 = nn.LayerNorm(v_in_dim) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(v_in_dim) if ln else nn.Identity()

        # multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=v_in_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # feed forward network
        self.fc_o = nn.Sequential(
            nn.Linear(v_in_dim, v_in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(v_in_dim, v_in_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_proj = self.fc_q(queries)
        k_proj = self.fc_k(keys)
        v_proj = self.fc_v(keys)

        # attention with residual connection and layer norm
        attn_out, _ = self.mha(
            query=q_proj, key=k_proj, value=v_proj, key_padding_mask=mask
        )
        updated_queries = self.ln0(q_proj + self.dropout(attn_out))

        # feed forward network with residual connection and layer norm
        return self.ln1(updated_queries + self.fc_o(updated_queries))


class SetAttnBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        ln: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mab = MultiheadAttnBlock(
            q_in_dim=in_dim,
            k_in_dim=in_dim,
            v_in_dim=out_dim,
            num_heads=num_heads,
            ln=ln,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mab(x, x, mask=mask)


class PoolingMultiheadAttnBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        num_seeds: int,
        ln: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.Tensor(1, num_seeds, in_dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MultiheadAttnBlock(
            q_in_dim=in_dim,
            k_in_dim=in_dim,
            v_in_dim=in_dim,
            num_heads=num_heads,
            ln=ln,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        seeds = self.seeds.repeat(batch_size, 1, 1)
        return self.mab(seeds, x, mask=mask)


class SetTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_sabs: int = 2,
        output_hidden_dims: tuple[int] = (8,),
        dropout: float = 0.1,
        ln: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        assert hidden_dim % num_heads == 0, (
            f"Hidden dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )

        # encoder consisting of SABs
        self.sabs = nn.ModuleList(
            [
                SetAttnBlock(
                    input_dim,
                    hidden_dim,
                    num_heads,
                    ln=ln,
                    dropout=dropout,
                )
            ]
            + [
                SetAttnBlock(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    ln=ln,
                    dropout=dropout,
                )
                for _ in range(num_sabs - 1)
            ]
        )

        # pooling layer
        self.pma = PoolingMultiheadAttnBlock(
            hidden_dim,
            num_heads,
            num_seeds=1,
            ln=ln,
            dropout=dropout,
        )

        dims = [hidden_dim] + list(output_hidden_dims)
        self.decoder = nn.Sequential(
            *[
                layer
                for d_in, d_out in zip(dims[:-1], dims[1:])
                for layer in (
                    nn.Linear(in_features=d_in, out_features=d_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            ],
            nn.Linear(in_features=dims[-1], out_features=output_dim),
        )

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n_batch, n_set, n_x_dim = inputs.shape
        assert n_x_dim == self.input_dim, f"Expected {self.input_dim}, got {n_x_dim}"

        mask = (
            mask.bool()
            if mask is not None
            else torch.ones(
                (n_batch, n_set),
                device=inputs.device,
                dtype=torch.bool,
            )
        )

        key_padding_mask = ~mask
        is_empty_set = key_padding_mask.all(dim=1)

        if is_empty_set.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[is_empty_set, 0] = False

        features = inputs
        for sab in self.sabs:
            features = sab(features, mask=key_padding_mask)

        features = self.pma(features, mask=key_padding_mask)
        features = features.squeeze(1)

        if is_empty_set.any():
            features = torch.where(
                is_empty_set.unsqueeze(1),
                torch.zeros_like(features),
                features,
            )

        return self.decoder(features)
