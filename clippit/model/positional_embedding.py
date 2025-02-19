import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_length: int, d_model: int):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model

        self.pos_embedding = nn.Parameter(
            torch.randn((self.seq_length, self.d_model)) * 0.02
        )

    def forward(self, x: torch.Tensor):
        """Receives input x and adds learnable positional embeddings to the input patches

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_length, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model)
        """

        batch_size, seq_length, d_model = x.shape

        if seq_length > self.seq_length:
            raise ValueError(
                f"Input Sequence length ({seq_length}) exceeds maximum length ({self.seq_length})"
            )

        pos_emb_trunc = self.pos_embedding[:seq_length, :]

        out = x + pos_emb_trunc

        return out
