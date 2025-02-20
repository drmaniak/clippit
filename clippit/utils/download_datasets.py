from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from PIL.JpegImagePlugin import JpegImageFile
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.tokenization_utils_base import BatchEncoding


def process_and_save_clip_embeddings(
    output_dir: Path | str,
    topk: int = 1,
    shortest_edge: int = 224,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Process Flickr30k dataset and save CLIP embeddings with topk similar captions.

    Args:
        output_dir: Directory to save the processed dataset
        split: Dataset split ('train', 'test', 'validation')
        topk: Number of most similar captions to keep per image
        batch_size: Batch size for processing
        device: Device to use for computation
    """
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )  # type: ignore

    # Load Flickr dataset
    flickr = load_dataset("nlphuji/flickr30k")
    dataset: DatasetDict = flickr["test"]  # type: ignore

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output file
    data_dict = {"train": [], "test": [], "val": []}
    chunk_count = 0
    rows_count = 0
    MAX_ROWS = 20000  # Maximum rows before saving to disk
    chunk_files = []  # Keep track of temporary chunk files

    # Process dataset in batches
    for row in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
        image: JpegImageFile = row["image"]
        captions: list[str] = row["caption"]
        split: str = row["split"]
        width, height = image.size
        image_id: int = int(row["img_id"])
        filename: str = row["filename"]

        # Pass the image & 5 captions to the CLIP Processor
        vision_input: BatchEncoding = processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": shortest_edge},
            padding=True,
        ).to(device)

        text_input: BatchEncoding = processor(
            text=captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(device)

        model_input: BatchEncoding = processor(
            text=captions,
            images=image,
            return_tensors="pt",
            size={"shortest_edge": shortest_edge},
            padding=True,
            truncation=True,
        ).to(device)

        # Pass this input into CLIP to get outputs
        model_output: CLIPOutput = model(**model_input)

        # This obtains the CLS token for the image (batch_size, d_model=512)
        image_output = model.get_image_features(**vision_input).squeeze()  # type: ignore

        # This obtains 77 seq length image for each caption shape (5, 77, d_model=512)
        text_output = model.text_model(**text_input).last_hidden_state

        # We will now pick the top-k most similar captions
        vals, caption_indices = model_output["logits_per_image"].topk(k=topk)
        for idx in caption_indices[0].tolist():
            caption_embedding = text_output[idx].tolist()
            attention_mask = text_input["attention_mask"][idx].tolist()  # type: ignore
            caption_tokens = text_input["input_ids"][idx].tolist()  # type: ignore
            data_row = {
                "img_embedding": image_output.tolist(),
                "caption_text": captions[idx],
                "caption_embedding": caption_embedding,
                "attention_mask": attention_mask,
                "caption_tokens": caption_tokens,
                "img_id": image_id,
                "filename": filename,
            }
            # Append the row to the data_list for the corresponding split
            data_dict[split].append(data_row)
            rows_count += 1

            # Save to disk if we hit the row limit
            if rows_count >= MAX_ROWS:
                print(f"\nSaving chunk {chunk_count} to disk...")
                for split_name in ["train", "test", "val"]:
                    if data_dict[split_name]:  # Only save if there's data
                        chunk_file = (
                            output_dir
                            / f"flickr_{split_name}_chunk_{chunk_count}.parquet"
                        )
                        chunk_files.append(chunk_file)
                        df = pd.DataFrame(data_dict[split_name])
                        df.to_parquet(chunk_file)
                        data_dict[split_name] = []  # Clear the list
                chunk_count += 1
                rows_count = 0

        torch.cuda.empty_cache()

    # Save any remaining data
    if any(len(data_dict[split]) > 0 for split in ["train", "test", "val"]):
        print(f"\nSaving final chunk {chunk_count} to disk...")
        for split_name in ["train", "test", "val"]:
            if data_dict[split_name]:
                chunk_file = (
                    output_dir / f"flickr_{split_name}_chunk_{chunk_count}.parquet"
                )
                chunk_files.append(chunk_file)
                df = pd.DataFrame(data_dict[split_name])
                df.to_parquet(chunk_file)

    # Merge all chunks into final files
    print("\nMerging chunks into final files...")
    for split_name in ["train", "test", "val"]:
        split_chunks = [
            f for f in chunk_files if f.name.startswith(f"flickr_{split_name}_chunk_")
        ]
        if split_chunks:
            dfs = [pd.read_parquet(chunk) for chunk in split_chunks]
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                final_file = output_dir / f"flickr_{split_name}_top{topk}.parquet"
                final_df.to_parquet(final_file)
                print(f"Created {final_file} with {len(final_df)} rows")

    # Clean up chunk files
    print("\nCleaning up temporary files...")
    for chunk_file in chunk_files:
        if chunk_file.exists():
            chunk_file.unlink()


def save_dataframe_parquet(
    data_dict: dict[str, Any], topk: int, split: str, output_dir: Path | str
):
    df = pd.DataFrame(data_dict[split])
    filepath = Path(output_dir) / f"flickr_{split}_top{topk}.parquet"
    df.to_parquet(filepath)
