import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Image, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


class Flicker30K(Dataset):
    # TODO: Try swapping out clip tokenizer & text embeddings for GPT2's
    def __init__(
        self, datafile: Path | str, clip_processor: CLIPProcessor, clip_model: CLIPModel
    ):
        super().__init__()
        datafile = Path(datafile)
        if datafile.is_file() and datafile.suffix == ".parquet":
            self.dataset = pd.read_parquet(datafile)
            self.processor = clip_processor
            self.model = clip_model
        else:
            raise FileNotFoundError(f"No datafile found in {datafile}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_emb, cap = self.dataset.iloc[idx][["img_embedding", "caption"]]

        img_emb = torch.tensor(img_emb)
        cap_processed = self.processor(
            text=[cap],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        cap_tokens = cap_processed["input_ids"].squeeze()  # size (77) # type: ignore
        cap_output = self.model.text_model(
            **cap_processed,
            output_hidden_states=True,
            return_dict=True,
        )
        attention_mask = cap_processed.attention_mask.squeeze(0)[:-1]  # (76,)
        cap_emb = cap_output.last_hidden_state.squeeze()  # size (77, 512)
        cap_emb_decoder_input = cap_emb[
            1:-1, :
        ].detach()  # Don't send the last token # size (75, 512)
        cap_emb_decoder_target = cap_tokens[
            1:
        ].detach()  # Don't include the start token # size (76)

        decoder_input = torch.cat(
            (img_emb.unsqueeze(0), cap_emb_decoder_input), dim=0
        )  # size (76, 512)
        target_output = cap_emb_decoder_target  # size (76, 512)

        return {
            "decoder_input": decoder_input.type(torch.float32),
            "target_output": target_output.type(torch.float32),
            "attention_mask": attention_mask.type(torch.float32),
        }
