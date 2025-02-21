import torch
import torch.nn as nn
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.tokenization_utils_base import BatchEncoding

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

    def forward(self, x: torch.Tensor, custom_attn_mask: torch.Tensor | None = None):
        """Receives input x and Query/Key values from Encoder and transforms it with an Decoder Block (Masked MultiHeadedAttention, Cross Attention, MLP, Residual Connection)

        Args:
            x (torch.Tensor): Input with dims (batch_size, seq_length, d_model)
            custom_attn_mask (torch.Tensor): Attention Mask for padded sequences (batch_size, seq_length)
        """

        # Sublayer 1
        residual = x
        x = self.layer_norm1(x)
        x = self.masked_self_attention(x, custom_attn_mask)
        if self.dropout:
            x = self.dropout_layer(x)
        x = x + residual

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
        input_dim: int,
        seq_length: int,
        d_model: int,
        num_decoder_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float | None = None,
        num_classes: int = 10,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_classes = num_classes

        # Initialize the layers

        # Project the input vector dimension to the model dimension
        self.projection_layer = nn.Linear(self.input_dim, self.d_model)

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
        print(f"Dtype: {tensor.dtype}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

    def forward(
        self, x: torch.Tensor, custom_attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input embedded sequence of shape (batch_size, seq_length=76, emb_dim=512).
            custom_attn_mask (torch.Tensor): Attention Mask for padded sequences (batch_size, seq_length)

        Process:
            1. Training Mode (teacher forcing):
               - Input sequence [CLS, x₁,x₂,x₃,x₄]
               - Model predicts [x₁,x₂,x₃,x₄, x_5]
               - All positions processed in parallel

        Returns:
            torch.Tensor: Logits for each position in sequence.
                         Shape: (batch_size, seq_length, num_classes)
        """
        if custom_attn_mask is not None:
            # print("\nDecoder Mask Debug Info:")
            # print(f"Input shape: {x.shape}")
            # print(f"Mask shape: {custom_attn_mask.shape}")
            # print(f"Mask dtype: {custom_attn_mask.dtype}")
            # print(f"Number of True values in mask: {custom_attn_mask.sum().item()}")
            # print(f"Percentage of True values: {(custom_attn_mask.sum() / custom_attn_mask.numel() * 100):.2f}%")

            if custom_attn_mask.dim() != 2:
                raise ValueError(f"Expected 2D mask, got {custom_attn_mask.dim()}D")
            if custom_attn_mask.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Mask batch size {custom_attn_mask.shape[0]} doesn't match input batch size {x.shape[0]}"
                )
            if custom_attn_mask.shape[1] < x.shape[1]:
                raise ValueError(
                    f"Mask sequence length {custom_attn_mask.shape[1]} is shorter than input sequence length {x.shape[1]}"
                )

            # Print sample of mask values
            # print("\nSample of mask values (first sequence):")
            # print(custom_attn_mask[0, :10])  # First 10 values of first sequence

        batch_size = x.shape[0]

        # Prepend start token to sequence, exlude last token
        input_tokens = x  # (batch_size, seq_length=76, d_model)

        input_tokens = self.projection_layer(
            input_tokens
        )  # Project to model dimension (batch_size, seq_length, d_model)

        # Add position embedding
        decoder_input = self.pos_embedding(
            input_tokens
        )  # (batch_size, seq_length, d_model)

        # Pass through decoder blocks
        decoder_output = decoder_input
        for i, decoder_block in enumerate(self.decoder_blocks):
            decoder_output = decoder_block(decoder_output, custom_attn_mask)

        # Project to num_classes for output prediction
        logits = self.output_projection(
            decoder_output
        )  # (batch_size, seq_length, num_classes)

        return logits

    def inference(
        self,
        img_emb: torch.Tensor | None,
        image: Image | None,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        top_k: int = 50,
        max_length: int | None = None,
        min_length: int = 15,
    ):
        """Generate a caption using the trained decoder model.

        Args:
            decoder_input: Optional pre-computed embeddings
            image: Optional input image to caption
            clip_model: CLIP model for computing embeddings
            clip_processor: CLIP processor for tokenization
            device: Device to run inference on
            temperature: Sampling temperature (higher = more diverse)
            top_k: Number of highest probability tokens to keep
            max_length: Maximum caption length
        """
        self.eval()

        with torch.no_grad():
            # Get initial sequence embedding
            if img_emb is not None:
                # Use provided embeddings (e.g. from training data)
                current_sequence = img_emb.unsqueeze(1).to(device)
            else:
                # Compute embeddings from image
                inputs = clip_processor(
                    images=image, return_tensors="pt", padding=True
                ).to(device)

                image_features = clip_model.get_image_features(**inputs)  # type: ignore
                current_sequence = image_features.unsqueeze(1)

            # Initialize generation
            max_length = max_length or self.seq_length
            generated_tokens = []

            # Get special token IDs
            bos_token_id, eos_token_id = clip_processor.tokenizer.all_special_ids  # type: ignore

            for step in range(max_length):
                # Create attention mask for generated sequence (all tokens are valid)
                seq_mask = torch.ones(
                    (1, current_sequence.shape[1]), device=device, dtype=torch.bool
                )

                # Get model predictions
                logits = self.forward(current_sequence, seq_mask)
                next_token_logits = logits[:, -1, :]

                # Apply temperature and length penalties
                next_token_logits = next_token_logits / temperature

                # Prevent EOS before min_length
                if step < min_length:
                    next_token_logits[:, eos_token_id] = float("-inf")

                # Add EOS penalty to encourage longer sequences
                next_token_logits[:, eos_token_id] -= 2.0

                # Filter special tokens
                next_token_logits[:, bos_token_id] = float("-inf")

                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                # Stop if EOS token
                if token_id == eos_token_id:
                    break

                generated_tokens.append(token_id)

                # Get embedding for next token
                token_inputs = torch.tensor([generated_tokens], device=device)
                token_outputs = clip_model.text_model(
                    token_inputs, output_hidden_states=True, return_dict=True
                )
                token_embedding = token_outputs.last_hidden_state

                # Update sequence
                current_sequence = torch.cat([current_sequence, token_embedding], dim=1)

                # Debug output
                if step % 5 == 0:
                    partial_caption = clip_processor.tokenizer.decode(generated_tokens)  # type: ignore
                    print(f"Step {step}: {partial_caption}")

            # Decode final caption
            caption = clip_processor.tokenizer.decode(generated_tokens)  # type: ignore
            print(f"\nFinal caption: {caption}")

            return caption, generated_tokens
