import torch
import torch.nn as nn


class MultiheadAttnBlock(nn.Module):
    def __init__(
        self,
        q_in_dim: int,
        k_in_dim: int,
        v_in_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_apply_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.v_in_dim = v_in_dim
        self.num_heads = num_heads

        # projections
        self.fc_q = nn.Linear(in_features=q_in_dim, out_features=v_in_dim)
        self.fc_k = nn.Linear(in_features=k_in_dim, out_features=v_in_dim)
        self.fc_v = nn.Linear(in_features=k_in_dim, out_features=v_in_dim)

        # layer norms
        if is_apply_layer_norm:
            self.norm_attn = nn.LayerNorm(v_in_dim)
            self.norm_ffn = nn.LayerNorm(v_in_dim)
        else:
            self.norm_attn = nn.Identity()
            self.norm_ffn = nn.Identity()

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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # queries: (batch, num_queries, q_in_dim)
        # keys: (batch, set_size, k_in_dim)
        # mask: (batch, set_size) - True for padded positions
        q_proj, k_proj, v_proj = self.fc_q(queries), self.fc_k(keys), self.fc_v(keys)

        attn_out, _ = self.mha(
            query=q_proj,
            key=k_proj,
            value=v_proj,
            key_padding_mask=mask,
        )
        updated_queries = self.norm_attn(q_proj + self.dropout(attn_out))

        # (batch, num_queries, v_in_dim)
        return self.norm_ffn(updated_queries + self.fc_o(updated_queries))


class SetAttnBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.mab = MultiheadAttnBlock(
            q_in_dim=in_dim,
            k_in_dim=in_dim,
            v_in_dim=out_dim,
            num_heads=num_heads,
            is_apply_layer_norm=is_apply_layer_norm,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch, set_size, in_dim)
        # mask: (batch, set_size) - True for padded positions
        # returns: (batch, set_size, out_dim)
        return self.mab(x, x, mask=mask)


class PoolingMultiheadAttnBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        num_seeds: int,
        dropout: float = 0.0,
        is_apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.Tensor(1, num_seeds, in_dim)) 
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MultiheadAttnBlock(
            q_in_dim=in_dim,
            k_in_dim=in_dim,
            v_in_dim=in_dim,
            num_heads=num_heads,
            is_apply_layer_norm=is_apply_layer_norm,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch, set_size, in_dim)
        # mask: (batch, set_size) - True for padded positions
        # returns: (batch, num_seeds, in_dim)
        batch_size = x.size(0)
        seeds = self.seeds.expand(batch_size, -1, -1)
        return self.mab(seeds, x, mask=mask)


class SetTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_sabs: int = 2,
        output_hidden_dims: int | list[int] | tuple[int, ...] | None = (8,),
        dropout: float = 0.1,
        is_apply_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
            )

        # encoder with stacked self-attention blocks
        self.sabs = nn.ModuleList(
            [
                SetAttnBlock(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    num_heads,
                    is_apply_layer_norm=is_apply_layer_norm,
                    dropout=dropout,
                )
                for i in range(num_sabs)
            ]
        )

        # pooling multi-head attention block
        self.pma = PoolingMultiheadAttnBlock(
            hidden_dim,
            num_heads,
            num_seeds=1,
            is_apply_layer_norm=is_apply_layer_norm,
            dropout=dropout,
        )

        # decoder MLP
        if output_hidden_dims is None:
            out_h = []
        elif isinstance(output_hidden_dims, int):
            out_h = [output_hidden_dims]
        else:
            out_h = list(output_hidden_dims)
        dims = [hidden_dim] + out_h + [output_dim]

        self.decoder = nn.Sequential(
            *[
                layer
                for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]))
                for layer in (
                    [nn.Linear(d_in, d_out)]
                    + ([nn.ReLU(), nn.Dropout(dropout)] if i < len(dims) - 2 else [])
                )
            ]
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # inputs: (batch, set_size, input_dim)
        # mask: (batch, set_size) - True for valid positions
        batch_size, set_size, feat_dim = inputs.shape
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}, got {feat_dim}")

        if mask is None:
            mask = torch.ones(
                (batch_size, set_size),
                device=inputs.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.bool()

        # transform mask to key_padding_mask (True for padded positions)
        key_padding_mask = ~mask

        # handle empty sets -> return zeros
        is_empty_set = key_padding_mask.all(dim=1)
        if is_empty_set.all():
            return torch.zeros(
                (batch_size, self.output_dim),
                device=inputs.device,
                dtype=inputs.dtype,
            )

        # unmask first element for partially empty sets to avoid attention issues
        if is_empty_set.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[is_empty_set, 0] = False

        # encode through self-attention blocks
        features = inputs
        for sab in self.sabs:
            features = sab(features, mask=key_padding_mask)

        # global aggregation: (batch, 1, hidden_dim) -> (batch, hidden_dim)
        features = self.pma(features, mask=key_padding_mask).squeeze(1)

        # zero out results for empty sets
        features = torch.where(
            is_empty_set.unsqueeze(1),
            torch.zeros_like(features),
            features,
        )

        # (batch, hidden_dim) -> (batch, output_dim)
        return self.decoder(features)
