# Set tokenizer parallelism to false to avoid warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import wandb
from torch._prims_common import FloatLike
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from clippit.data.dataset import Flicker30K
from clippit.model.decoder import Decoder


def debug_tensor(name: str, tensor: torch.Tensor, batch_idx: int) -> None:
    print(f"\nDebug {name} (Batch {batch_idx}):")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")
    print(f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
    if torch.isnan(tensor).any():
        print("WARNING: Contains NaN values")
    if torch.isinf(tensor).any():
        print("WARNING: Contains Inf values")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Custom Decoder to generate captions for Flickr30K images"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_dataloaders(
    config: Dict[str, Any], clip_model: CLIPModel, clip_processor: CLIPProcessor
):
    train_dataset = Flicker30K(
        datafile=Path(config["data"]["flickr_train_path"]),
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    val_dataset = Flicker30K(
        datafile=Path(config["data"]["flickr_val_path"]),
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    # val_dataset = Flicker30K(
    #     datafile=Path(config["data"]["flickr_test_path"]),
    #     clip_model=clip_model,
    #     clip_processor=clip_processor,
    # )

    data_dict = train_dataset[0]
    decoder_input = data_dict["decoder_input"]
    target_output = data_dict["target_output"]
    attention_mask = data_dict["attention_mask"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, (decoder_input, target_output, attention_mask)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_processor: CLIPProcessor,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch}",
        leave=False,
    )

    for batch_idx, batch_dict in enumerate(progress_bar):
        decoder_input = batch_dict["decoder_input"].to(device)
        target_output = batch_dict["target_output"].to(device)
        attention_mask = batch_dict["attention_mask"].to(device)

        optimizer.zero_grad()

        try:
            # Forward pass
            outputs = model(
                decoder_input, attention_mask
            )  # shape: (batch_size, seq_len, vocab_size)

            # Reshape for loss calculation
            batch_size, seq_length, num_classes = outputs.shape
            outputs = outputs.view(-1, num_classes).to(torch.float32)
            labels = target_output.view(-1).to(torch.long)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Debug info
            if batch_idx % 100 == 0:
                pred_tokens = outputs.argmax(dim=-1)[:10]  # First 10 predictions
                true_tokens = labels[:10]
                pred_text = clip_processor.tokenizer.decode(pred_tokens)  # type: ignore
                true_text = clip_processor.tokenizer.decode(true_tokens)  # type: ignore
                wandb.log(
                    {
                        "predictions": wandb.Table(
                            data=[[pred_text, true_text]], columns=["Predicted", "True"]
                        )
                    }
                )

            # Check for NaN/Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"\n=== NaN/Inf detected in loss at batch {batch_idx} ===")
                debug_tensor("outputs", outputs, batch_idx)
                debug_tensor("labels", labels, batch_idx)
                raise ValueError("NaN/Inf in loss")

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print(
                f"Shapes - Input: {decoder_input.shape}, Target: {target_output.shape}"
            )
            raise e

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != criterion.ignore_index
        correct_predictions += ((predictions == labels) & mask).sum().item()
        total_predictions += mask.sum().item()
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_predictions / max(1, total_predictions):.2f}%",
                "perplexity": f"{torch.exp(torch.tensor(loss.item())):.2f}",
            }
        )

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / max(1, total_predictions)
    epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

    return epoch_loss, epoch_accuracy, epoch_perplexity.item()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    clip_processor: CLIPProcessor,
):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for batch_idx, batch_dict in enumerate(progress_bar):
        decoder_input = batch_dict["decoder_input"].to(device)
        target_output = batch_dict["target_output"].to(device)
        attention_mask = batch_dict["attention_mask"].to(device)

        # Forward pass with teacher forcing for validation
        outputs = model(decoder_input, attention_mask)

        # Reshape for loss calculation
        batch_size, seq_length, num_classes = outputs.shape
        outputs = outputs.view(-1, num_classes).to(torch.float32)
        labels = target_output.view(-1).to(torch.long)

        loss = criterion(outputs, labels.view(-1))

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\n=== NaN/Inf detected in loss at batch {batch_idx} ===")
            raise ValueError("NaN/Inf in loss")

        if batch_idx % 100 == 0:
            pred_tokens = outputs.argmax(dim=-1)[:10]  # First 10 predictions
            true_tokens = labels[:10]
            pred_text = clip_processor.tokenizer.decode(pred_tokens)  # type: ignore
            true_text = clip_processor.tokenizer.decode(true_tokens)  # type: ignore
            wandb.log(
                {
                    "predictions": wandb.Table(
                        data=[[pred_text, true_text]], columns=["Predicted", "True"]
                    )
                }
            )

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != criterion.ignore_index  # Handle padding tokens
        correct_predictions += ((predictions == labels) & mask).sum().item()
        total_predictions += mask.sum().item()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_predictions / total_predictions:.2f}%",
            }
        )

    val_loss = total_loss / len(val_loader)
    val_accuracy = 100 * correct_predictions / total_predictions

    val_perplexity = torch.exp(torch.tensor(val_loss))

    return val_loss, val_accuracy, val_perplexity.item()


def main():
    args = parse_args()
    config = load_config(args.config)

    required_params = {
        "model": [
            "d_model",
            "num_decoder_blocks",
            "num_heads",
            "mlp_ratio",
            "dropout",
        ],
        "training": [
            "batch_size",
            "learning_rate",
            "weight_decay",
            "num_epochs",
            "checkpoint_dir",
        ],
        "data": [
            "flickr_train_path",
            "flickr_val_path",
            "flickr_test_path",
            "num_workers",
        ],
        "wandb": ["project_name", "run_name"],
    }

    for section, params in required_params.items():
        assert section in config, f"Missing section: {section}"
        for param in params:
            assert param in config[section], f"Missing parameter: {param} in {section}"

    # Initialize wandb
    wandb.init(
        project=f"{config['wandb']['project_name']}",
        config=config,
    )

    # Create checkpoint directory
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize pretrained CLIPModel & CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )  # type: ignore

    # Create dataloaders
    train_loader, val_loader, (decoder_input, target_output, attention_mask) = (
        create_dataloaders(config, clip_model, clip_processor)
    )
    print(f"Train dataset size: {len(train_loader.dataset)}")  # type: ignore
    print(f"Validation dataset size: {len(val_loader.dataset)}")  # type: ignore

    # Add assertions to verify data sample shapes
    assert decoder_input.dim() == 2, f"Expected 2D tensor for data_sample, got {decoder_input.dim()}D"  # fmt: skip
    assert target_output.dim() == 1, f"Expected 1D tensor for label_sample, got {target_output.dim()}D"  # fmt: skip
    print(f"Data sample shape: {decoder_input.shape}, Label sample shape: {target_output.shape}")  # fmt: skip

    seq_length, input_dim = decoder_input.shape
    num_classes = len(clip_processor.tokenizer.get_vocab())  # type: ignore

    # Initialize model
    model = Decoder(
        input_dim=input_dim,
        seq_length=seq_length,
        d_model=config["model"]["d_model"],
        num_decoder_blocks=config["model"]["num_decoder_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        num_classes=num_classes,
    ).to(device)

    # After model initialization:
    print("\nModel Architecture:")
    print(f"Sequence Length: {seq_length}")
    print(f"Seq dimension: {input_dim}")
    print(f"Hidden dimension: {config['model']['d_model']}")
    print(f"Number of decoder blocks: {config['model']['num_decoder_blocks']}")

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Initialize criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    num_warmup_steps = num_training_steps // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        total_steps=num_training_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    # After optimizer initialization:
    wandb.watch(model, criterion, log="all", log_freq=100)

    # Training loop
    best_val_accuracy = 0
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_loss, train_accuracy, train_perplexity = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, clip_processor
        )

        # Validate
        val_loss, val_accuracy, val_perplexity = validate(
            model, val_loader, criterion, device, clip_processor
        )

        # Enhanced logging
        wandb.log(
            {
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "train/perplexity": train_perplexity,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "val/perplexity": val_perplexity,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
                "gradient_norm": torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                ).item(),
            }
        )

        # Log sample predictions periodically
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_input = sample_batch["decoder_input"][:1].to(device)
                sample_output = model.inference(
                    decoder_input=sample_input,
                    image=None,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                )
                sample_text = clip_processor.tokenizer.decode(sample_output)  # type: ignore
                wandb.log(
                    {
                        "sample_generations": wandb.Table(
                            data=[[sample_text]], columns=["Generated Text"]
                        )
                    }
                )
            model.train()
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "config": config,
                },
                f"{config['training']['checkpoint_dir']}/best_model.pth",
            )
            print(f"Saved new best model with validation accuracy: {val_accuracy:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
