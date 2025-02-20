import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int | None, masked: bool = False):
        super().__init__()

        # Assign instance vars
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v if d_v else self.d_k
        self.masked = masked

        # Create Layers
        self.W_q = nn.Linear(in_features=d_model, out_features=self.d_k, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=self.d_k, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=self.d_v, bias=False)

    def forward(self, x: torch.Tensor, custom_attn_mask: torch.Tensor | None = None):
        """Receives input x and returns Attention Weighted Vectors

        Args:
            x (torch.Tensor): Input with dims (batch_size, seq_length, d_model)
            custom_attn_mask (torch.Tensor): Attention Mask for padded sequences (batch_size, seq_length)
        """

        # Obtain Projections
        Q = self.W_q(x)  # (batch_size, num_patches, d_k)
        K = self.W_k(x)  # (batch_size, num_patches, d_k)
        V = self.W_v(x)  # (batch_size, num_patches, d_v)

        # Compute Attention Weights
        scaling_factor = torch.sqrt(
            torch.tensor(self.d_k, device=x.device, dtype=x.dtype)
        )
        scaled_attention_map = (
            torch.einsum("bqd,bkd->bqk", Q, K) / scaling_factor
        )  # (batch_size, num_patches, num_patches)

        # TODO: Add flow to separate causal and padding masks
        if self.masked:
            causal_attention_mask = self.compute_attention_mask(scaled_attention_map)
            if custom_attn_mask is not None:
                # Convert mask to proper shape: [batch_size, 1, seq_length]
                # and expand to [batch_size, seq_length, seq_length]
                seq_length = scaled_attention_map.shape[1]
                padding_mask = (
                    custom_attn_mask[:, :seq_length]
                    .bool()
                    .to(scaled_attention_map.device)
                )
                padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_length, -1)

                # Combine masks:
                # - causal_attention_mask prevents looking ahead
                # - padding_mask allows attending only to valid tokens
                attention_mask = causal_attention_mask | (~padding_mask)

                # Debug mask information
                # print("\nMask Debug Info:")
                # print(f"Causal mask shape: {causal_attention_mask.shape}")
                # print(f"Padding mask shape: {padding_mask.shape}")
                # print(f"Combined mask shape: {attention_mask.shape}")
                # print(f"Attention map shape: {scaled_attention_map.shape}")
                # print(
                #     f"Number of True values in causal mask: {causal_attention_mask.sum().item()}"
                # )
                # print(
                #     f"Number of True values in padding mask: {padding_mask.sum().item()}"
                # )
                # print(
                #     f"Number of True values in combined mask: {attention_mask.sum().item()}"
                # )
                # print(
                #     f"Sample of attention weights before masking:\n{scaled_attention_map[0, :15, :15]}"
                # )

                # Apply mask (mask out with -inf where attention_mask is True)
                scaled_attention_map = scaled_attention_map.masked_fill(
                    attention_mask, float("-inf")
                )

                # print(
                #     f"Sample of attention weights after masking:\n{scaled_attention_map[0, :15, :15]}"
                # )

        attention_weights = F.softmax(
            scaled_attention_map, dim=-1
        )  # Softmax over the "Key" dimension (columns)

        # Compute Attention Weighted Values
        attention_weighted_value = torch.einsum(
            "bqk,bkd->bqd", attention_weights, V
        )  # (batch_size, num_patches, d_v)

        assert attention_weighted_value.shape == (x.shape[0], x.shape[1], self.d_v)

        return attention_weighted_value

    def compute_attention_mask(
        self, scaled_attention_map: torch.Tensor
    ) -> torch.Tensor:
        causal_attention_mask = torch.triu(
            torch.ones_like(scaled_attention_map, dtype=torch.bool), diagonal=1
        ).to(scaled_attention_map.device)

        return causal_attention_mask


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        h: int,
        dropout: float | None = None,
        masked: bool = False,
    ):
        super().__init__()

        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = self.d_k  # By doing this, I lose flexibility in setting d_v
        self.masked = masked

        # Create h AttentionHeads
        self.attention_heads = nn.ModuleList(
            [
                SelfAttentionHead(self.d_model, self.d_k, self.d_v, masked=self.masked)
                for _ in range(self.h)
            ]
        )

        # Create a layer to project it back to d_model (parameterized transformation!)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor, custom_attn_mask: torch.Tensor | None = None):
        """Receives input x and returns Attention Weighted Vectors

        Args:
            x (torch.Tensor): Input with dims (batch_size, num_patches, d_model)
            custom_attn_mask (torch.Tensor): Attention Mask for padded sequences (batch_size, seq_length)
        """

        # NOTE: It is possible to parallelize obtaining the outputs from the attention heads

        # Obtain Attention Head Outputs from each head
        attention_head_outputs = [
            head(x, custom_attn_mask) for head in self.attention_heads
        ]  # (batch_size, num_patches, d_v)

        # Concatenate the outputs along the last dimension
        combined_attention = torch.cat(
            attention_head_outputs, dim=-1
        )  # (batch_size, num_patches, d_model)

        # Project it linearly to allow interaction between attention head outputs
        output = self.W_o(combined_attention)  # (batch_size, num_patches, d_model)

        # Apply dropout if present
        if self.dropout:
            output = self.dropout(output)

        return output
