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
        )

        model_input: BatchEncoding = processor(
            text=captions,
            images=image,
            return_tensors="pt",
            size={"shortest_edge": shortest_edge},
            padding=True,
        ).to(device)

        # Pass this input into CLIP to get outputs
        model_output: CLIPOutput = model(**model_input)

        # This obtains the CLS token for the image (batch_size, d_model=512)
        image_output = model.get_image_features(**vision_input).squeeze()  # type: ignore

        # We will now pick the top-k most similar captions
        vals, caption_indices = model_output["logits_per_image"].topk(k=topk)
        for idx in caption_indices[0].tolist():
            data_row = {
                "img_embedding": image_output.tolist(),
                "caption": captions[idx],
                "img_id": image_id,
                "filename": filename,
            }
            # Append the row to the data_list for the corresponding split
            data_dict[split].append(data_row)

    # Once done with making lists, create dataframes, save as parquet
    save_dataframe_parquet(
        data_dict=data_dict, topk=topk, split="train", output_dir=output_dir
    )
    save_dataframe_parquet(
        data_dict=data_dict, topk=topk, split="val", output_dir=output_dir
    )
    save_dataframe_parquet(
        data_dict=data_dict, topk=topk, split="test", output_dir=output_dir
    )


def save_dataframe_parquet(
    data_dict: dict[str, Any], topk: int, split: str, output_dir: Path | str
):
    df = pd.DataFrame(data_dict[split])
    filepath = Path(output_dir) / f"flickr_{split}_top{topk}.parquet"
    df.to_parquet(filepath)
