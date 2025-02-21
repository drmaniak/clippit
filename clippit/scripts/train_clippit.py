# Set tokenizer parallelism to false to avoid warnings
import os

from torch.optim.lr_scheduler import OneCycleLR
from wandb.data_types import Table

from clippit.model.clippit import ClippitModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from clippit.data.dataset import Flicker30K


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
    config: Dict[str, Any],
):
    train_dataset = Flicker30K(
        datafile=Path(config["data"]["flickr_train_path"]),
    )

    val_dataset = Flicker30K(
        datafile=Path(config["data"]["flickr_val_path"]),
    )

    data_dict = train_dataset[0]
    image_emb = data_dict["image_emb"]
    caption = data_dict["caption"]

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
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, (image_emb, caption)


def train_epoch(
    model: ClippitModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
    epoch: int,
    clip_processor: CLIPProcessor,
    train_predictions_table: Table,
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
        image_emb = batch_dict["image_emb"].to(device)
        caption = batch_dict["caption"]

        optimizer.zero_grad()
        target_output = torch.Tensor([])

        try:
            # Forward pass
            outputs, target_output, attention_mask = model(
                image_emb, caption
            )  # shape: (batch_size, seq_len, vocab_size)

            # Reshape for loss calculation
            batch_size, seq_length, num_classes = outputs.shape
            outputs = outputs.view(-1, num_classes).to(torch.float32)
            labels = target_output.reshape(-1).to(torch.long)

            # Calculate loss
            loss = criterion(outputs, labels)

            loss_per_token = loss.view(attention_mask.shape)
            masked_loss = loss_per_token * attention_mask.float()
            final_loss = masked_loss.sum() / attention_mask.sum()

            # Debug info
            if batch_idx % 50 == 0:
                pred_tokens = outputs.argmax(dim=-1)[:25]  # First 10 predictions
                true_tokens = labels[:25]
                pred_text = clip_processor.tokenizer.decode(pred_tokens)  # type: ignore
                true_text = clip_processor.tokenizer.decode(true_tokens)  # type: ignore
                if wandb.run is not None:
                    train_predictions_table.add_data(batch_idx, pred_text, true_text)  # type: ignore
                    wandb.log({"predictions_train": train_predictions_table})

            # Check for NaN/Inf
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"\n=== NaN/Inf detected in loss at batch {batch_idx} ===")
                debug_tensor("outputs", outputs, batch_idx)
                debug_tensor("labels", labels, batch_idx)
                raise ValueError("NaN/Inf in loss")

            final_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            print(f"Shapes - Input: {image_emb.shape}, Target: {target_output.shape}")
            raise e

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != criterion.ignore_index
        correct_predictions += ((predictions == labels) & mask).sum().item()
        total_predictions += mask.sum().item()
        total_loss += final_loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{final_loss.item():.4f}",
                "acc": f"{100 * correct_predictions / max(1, total_predictions):.2f}%",
                "perplexity": f"{torch.exp(torch.tensor(final_loss.item())):.2f}",
            }
        )

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / max(1, total_predictions)
    epoch_perplexity = torch.exp(torch.tensor(epoch_loss))

    return epoch_loss, epoch_accuracy, epoch_perplexity.item()


@torch.no_grad()
def validate(
    model: ClippitModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    clip_processor: CLIPProcessor,
    val_predictions_table: Table,
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
        image_emb = batch_dict["image_emb"].to(device)
        caption = batch_dict["caption"]

        # Forward pass with teacher forcing for validation
        outputs, target_output, attention_mask = model(image_emb, caption)

        # Reshape for loss calculation
        batch_size, seq_length, num_classes = outputs.shape
        outputs = outputs.view(-1, num_classes).to(torch.float32)
        labels = target_output.reshape(-1).to(torch.long)

        loss = criterion(outputs, labels.view(-1))

        loss_per_token = loss.view(attention_mask.shape)
        masked_loss = loss_per_token * attention_mask.float()
        final_loss = masked_loss.sum() / attention_mask.sum()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\n=== NaN/Inf detected in loss at batch {batch_idx} ===")
            raise ValueError("NaN/Inf in loss")

        if batch_idx % 15 == 0:
            pred_tokens = outputs.argmax(dim=-1)[:20]  # First 10 predictions
            true_tokens = labels[:20]
            pred_text = clip_processor.tokenizer.decode(pred_tokens)  # type: ignore
            true_text = clip_processor.tokenizer.decode(true_tokens)  # type: ignore

            if wandb.run is not None:
                val_predictions_table.add_data(batch_idx, pred_text, true_text)
                wandb.log({"predictions_val": val_predictions_table})

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != criterion.ignore_index  # Handle padding tokens
        correct_predictions += ((predictions == labels) & mask).sum().item()
        total_predictions += mask.sum().item()

        total_loss += final_loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{final_loss.item():.4f}",
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

    # Initialize wandb and create tables
    run = wandb.init(
        project=f"{config['wandb']['project_name']}",
        config=config,
    )

    # Create tables for logging predictions
    train_predictions_table: Table = wandb.Table(columns=["Step", "Predicted", "True"])
    val_predictions_table: Table = wandb.Table(columns=["Step", "Predicted", "True"])
    sample_generations_table: Table = wandb.Table(
        columns=[
            "Step",
            "Inference Caption",
            "Forward Pass Caption",
            "Ground Truth Caption",
        ]
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
    train_loader, val_loader, (img_embedding, caption) = create_dataloaders(config)
    print(f"Train dataset size: {len(train_loader.dataset)}")  # type: ignore
    print(f"Validation dataset size: {len(val_loader.dataset)}")  # type: ignore

    # seq_length, input_dim = decoder_input.shape
    seq_length = 76
    input_dim = 512
    num_classes = len(clip_processor.tokenizer.get_vocab())  # type: ignore

    # Initialize model
    model = ClippitModel(
        input_dim=512,
        seq_length=76,
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

    # Initialize criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(reduction="none")
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
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch + 1,
            clip_processor,
            train_predictions_table,
        )

        # Validate
        val_loss, val_accuracy, val_perplexity = validate(
            model, val_loader, criterion, device, clip_processor, val_predictions_table
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
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_img_emb = sample_batch["image_emb"][:1].to(
                    device
                )  # Take first image
                sample_caption = [sample_batch["caption"][0]]  # Take first caption

                # Forward pass with teacher forcing for validation
                outputs, target_output, attention_mask = model(
                    sample_img_emb, sample_caption
                )

                # Get model predictions
                pred_tokens = outputs[0].argmax(dim=-1).cpu()  # Take first sequence
                true_tokens = target_output[0].cpu()  # Take first sequence

                # Decode tokens to text
                forward_caption = clip_processor.tokenizer.decode(  # type: ignore
                    pred_tokens,
                    skip_special_tokens=True,
                )
                target_caption = clip_processor.tokenizer.decode(  # type: ignore
                    true_tokens,
                    skip_special_tokens=True,
                )

                print(f"Forward Generated Caption: {forward_caption}")

                # Get inference caption (if you have an inference method)
                sample_caption, sample_tokens = model.inference(
                    img_emb=sample_img_emb,
                    image=None,
                    max_length=50,
                    min_length=12,
                )
                if wandb.run is not None:
                    sample_generations_table.add_data(
                        epoch,
                        sample_caption,
                        (
                            forward_caption[0]
                            if isinstance(forward_caption, list)
                            else forward_caption
                        ),
                        (
                            target_caption[0]
                            if isinstance(target_caption, list)
                            else target_caption
                        ),
                    )
                    wandb.log({"sample_generations": sample_generations_table})

            model.train()
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = Path(config["training"]["checkpoint_dir"])
            best_model_path = (
                checkpoint_path
                / f"best_model_acc{val_accuracy:.2f}_epoch{epoch + 1}.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "train_perplexity": train_perplexity,
                    "val_perplexity": val_perplexity,
                    "config": config,
                    "best_val_accuracy": best_val_accuracy,
                },
                best_model_path,
            )
            print(f"Saved new best model with validation accuracy: {val_accuracy:.2f}%")

            # Create a symlink to the latest best model
            latest_best_link = checkpoint_path / "best_model.pth"
            if latest_best_link.exists():
                latest_best_link.unlink()
            latest_best_link.symlink_to(best_model_path.name)

        # Save final model at the end of training
        final_model_path = (
            Path(config["training"]["checkpoint_dir"])
            / f"final_model_epoch{config['training']['num_epochs']}.pth"
        )
        torch.save(
            {
                "epoch": config["training"]["num_epochs"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "train_perplexity": train_perplexity,
                "val_perplexity": val_perplexity,
                "config": config,
                "best_val_accuracy": best_val_accuracy,
            },
            final_model_path,
        )
        print(f"\nSaved final model after {config['training']['num_epochs']} epochs")

        # Log final metrics to wandb
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "best_val_accuracy": best_val_accuracy,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "final_train_accuracy": train_accuracy,
                    "final_val_accuracy": val_accuracy,
                    "final_train_perplexity": train_perplexity,
                    "final_val_perplexity": val_perplexity,
                }
            )

        wandb.finish()


if __name__ == "__main__":
    main()
