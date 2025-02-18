import torch
import torch.nn as nn

# from .cross_attention import MultiHeadedCrossAttention
from .positional_embedding import PositionalEmbedding
from .self_attention import MultiHeadedSelfAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float | None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Sublayer 1
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.masked_self_attention = MultiHeadedSelfAttention(
            self.d_model, self.num_heads, self.dropout, masked=True
        )

        # Sublayer 2
        # self.layer_norm2 = nn.LayerNorm(d_model)
        # self.cross_attention = MultiHeadedCrossAttention(
        #     self.d_model, self.num_heads, self.dropout
        # )

        # Sublayer 3
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.mlp = self._build_mlp_layer()

        if self.dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

    def _debug_tensor(self, name: str, tensor: torch.Tensor):
        """Helper to debug tensor values"""
        print(f"\n=== {name} ===")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

    def forward(self, x: torch.Tensor):
        """Receives input x and Query/Key values from Encoder and transforms it with an Decoder Block (Masked MultiHeadedAttention, Cross Attention, MLP, Residual Connection)

        Args:
            x (torch.Tensor): Input with dims (batch_size, seq_length, d_model)
        """

        # Sublayer 1
        residual = x
        x = self.layer_norm1(x)
        x = self.masked_self_attention(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = x + residual

        # # Sublayer 2
        # residual = x
        # x = self.layer_norm2(x)
        # x = self.cross_attention(x, encoder_output)
        # if self.dropout:
        #     x = self.dropout_layer(x)
        # x = x + residual

        # Sublayer 3
        residual = x
        x = self.layer_norm3(x)
        x = self.mlp(x)
        if self.dropout:
            x = self.dropout_layer(x)
        output = x + residual

        return output

    def _build_mlp_layer(self) -> nn.Sequential:
        hidden_dimension = int(self.d_model * (1 + self.mlp_ratio))
        mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, self.d_model),
        )
        return mlp


class Decoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        d_model: int,
        num_decoder_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float | None = None,
        num_classes: int = 10,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_classes = num_classes

        # Initialize the layers

        # Token Embedding to convert digit indices (0-9) to vectors
        self.token_embedding = nn.Embedding(
            self.num_classes + 1, embedding_dim=self.d_model
        )

        # Positional embedding for sequence positions
        self.pos_embedding = PositionalEmbedding(
            seq_length=self.seq_length, d_model=self.d_model
        )

        # Initialize the Decoder Blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(self.d_model, self.num_heads, self.mlp_ratio, self.dropout)
                for _ in range(self.num_decoder_blocks)
            ]
        )

        self.output_projection = nn.Linear(self.d_model, self.num_classes)

    def _debug_tensor(self, name: str, tensor: torch.Tensor):
        """Helper to debug tensor values"""
        print(f"\n=== {name} ===")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

    def forward(self, x: torch.Tensor, training_mode: bool = True):
        """Forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input digit indices of shape (batch_size, seq_length=76, emb_dim=512).
            training_mode (bool): If True, uses teacher forcing with parallel processing.
                            If False, generates sequence autoregressively.

        Process:
            1. Training Mode (teacher forcing):
               - Input sequence [x₁,x₂,x₃,x₄] becomes [START,x₁,x₂,x₃]
               - Model predicts [x₁,x₂,x₃,x₄]
               - All positions processed in parallel

            2. Inference Mode:
               - Starts with [START]
               - Autoregressively generates each next token
               - Uses previous predictions as context

        Returns:
            torch.Tensor: Logits for each position in sequence.
                         Shape: (batch_size, seq_length, num_classes)
                         Each position contains scores for all possible digits [0-9].
        """

        batch_size = x.shape[0]

        # Teacher forcing
        if training_mode:
            # Prepend start token to sequence, exlude last token
            start_token = (
                torch.full((batch_size, 1), self.num_classes).long().to(x.device)
            )
            input_tokens = torch.cat((start_token, x[:, :-1]), dim=1)

            # Embed the modified sequence
            decoder_input = self.token_embedding(
                input_tokens
            )  # (batch_size, seq_length, d_model)

            # Add position embedding
            decoder_input = self.pos_embedding(
                decoder_input
            )  # (batch_size, seq_length, d_model)

            # Pass through decoder blocks
            decoder_output = decoder_input
            for i, decoder_block in enumerate(self.decoder_blocks):
                decoder_output = decoder_block(decoder_output)

            # Project to num_classes for output prediction
            logits = self.output_projection(
                decoder_output
            )  # (batch_size, seq_length, num_classes)

            return logits

        else:
            # Inference mode
            # Autoregressive Generation
            batch_size = x.shape[0]
            start_token = (
                torch.full((batch_size, 1), self.num_classes).long().to(x.device)
            )
            current_sequence = start_token  # Start with just the start_token (zeros)
            outputs = []

            for _ in range(self.seq_length):
                # Embed current sequence
                decoder_input = self.token_embedding(
                    current_sequence
                )  # (batch_size, current_seq_length, d_model)
                # Add positional embeddings
                decoder_input = self.pos_embedding(
                    decoder_input
                )  # (batch_size, current_seq_length, d_model)

                # Pass through decoder blocks
                decoder_output = decoder_input
                for decoder_block in self.decoder_blocks:
                    decoder_output = decoder_block(
                        decoder_output
                    )  # (batch_size, current_seq_length, d_model)

                # Get next token prediction
                next_token_logits = self.output_projection(
                    decoder_output[:, -1:, :]
                )  # (batch_size, 1, num_classes)
                outputs.append(next_token_logits)

                if len(outputs) < self.seq_length:
                    # Use predicted token for next sequence
                    next_token = next_token_logits.argmax(dim=-1)  # (batch_size, 1)

                    current_sequence = torch.cat([current_sequence, next_token], dim=1)

            logits = torch.cat(outputs, dim=1)  # (batch_size, seq_length, num_classes)
            return logits
