import torch
import torch.nn as nn
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

from clippit.model.decoder import Decoder


class ClippitModel(nn.Module):
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        super().__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_classes = num_classes

        # Initialize the pretrained CLIP model for embedding captions and creating inputs for the decoder
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device  # type: ignore
        )
        self.clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )  # type: ignore

        self.decoder = Decoder(
            self.input_dim,
            self.seq_length,
            self.d_model,
            self.num_decoder_blocks,
            self.num_heads,
            self.mlp_ratio,
            self.dropout,
            self.num_classes,
        )

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
        self,
        img_embedding: torch.Tensor,
        caps: list[str],  # Now takes a list of captions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass through the Decoder.

        Args:
            img_embedding (torch.Tensor): Image Embedding (batch_size, emb_dim=512).
            cap (str): String caption for the image

        Process:
            1. Training Mode (teacher forcing):
               - Input sequence [CLS, x₁,x₂,x₃,x₄]
               - Model predicts [x₁,x₂,x₃,x₄, x_5]
               - All positions processed in parallel

        Returns:
            torch.Tensor: Logits for each position in sequence.
                         Shape: (batch_size, seq_length, num_classes)
        """
        pass

        decoder_input, attention_mask, target_tokens = self.prepare_data(
            img_embedding, caps
        )

        logits = self.decoder(
            decoder_input, attention_mask
        )  # (batch_size, seq_length, num_classes)

        return logits, target_tokens, attention_mask

    def prepare_data(
        self,
        img_embedding: torch.Tensor,  # (batch_size, 512)
        caps: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size = img_embedding.size(0)

        # Ensure img_embedding is on the correct device
        img_emb = img_embedding.to(self.device).unsqueeze(1)  # (batch_size, 1, 512)

        # Process all captions in batch
        cap_processed = self.clip_processor(
            text=caps,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
        )

        # Move processed data to device
        cap_processed = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in cap_processed.items()
        }

        # Get token IDs (batch_size, 77)
        cap_tokens = cap_processed["input_ids"]  # type: ignore

        # Process through CLIP text encoder
        cap_output = self.clip_model.text_model(
            **cap_processed,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get attention masks (batch_size, 76)
        attention_mask = cap_processed["attention_mask"][:, :-1]

        # Get embeddings (batch_size, 77, 512)
        cap_emb = cap_output.last_hidden_state

        # Remove first and last tokens for decoder input (batch_size, 75, 512)
        cap_emb_decoder_input = cap_emb[:, 1:-1, :].detach()

        # Remove first token for target (batch_size, 76)
        cap_emb_decoder_target = cap_tokens[:, 1:].detach()

        # Concatenate image embedding with caption embeddings
        # img_emb: (batch_size, 1, 512)
        # cap_emb_decoder_input: (batch_size, 75, 512)
        # Result: (batch_size, 76, 512)
        decoder_input = torch.cat((img_emb, cap_emb_decoder_input), dim=1)

        return decoder_input, attention_mask, cap_emb_decoder_target

    def inference(
        self,
        img_emb: torch.Tensor | None,
        image: Image | None,
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
            caption, generated_tokens = self.decoder.inference(
                img_emb,
                image,
                self.clip_model,
                self.clip_processor,
                device,
                temperature,
                top_k,
                max_length,
                min_length,
            )

            return caption, generated_tokens
