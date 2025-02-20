from pathlib import Path

import pandas as pd
import torch
from datasets import Image, load_dataset
from torch.utils.data import Dataset


class Flicker30K(Dataset):
    # TODO: Try swapping out clip tokenizer & text embeddings for GPT2's
    def __init__(self, datafile: Path | str):
        super().__init__()
        datafile = Path(datafile)
        if datafile.is_file() and datafile.suffix == ".parquet":
            self.dataset = pd.read_parquet(datafile)
        else:
            raise FileNotFoundError(f"No datafile found in {datafile}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_emb, cap, cap_emb, attention_mask, cap_tokens = self.dataset.iloc[idx][
            [
                "img_embedding",
                "caption_text",
                "caption_embedding",
                "attention_mask",
                "caption_tokens",
            ]
        ]

        img_emb = torch.tensor(img_emb)
        cap_emb = torch.tensor(cap_emb)
        attention_mask = torch.tensor(attention_mask)
        cap_tokens = torch.tensor(cap_tokens)

        attention_mask = attention_mask.squeeze()[:-1]  # (76,)
        cap_emb_decoder_input = cap_emb[
            1:-1, :
        ]  # Don't send the last token # size (75, 512)
        cap_emb_decoder_target = cap_tokens[
            1:
        ]  # Don't include the start token # size (76)

        decoder_input = torch.cat(
            (img_emb.unsqueeze(0), cap_emb_decoder_input), dim=0
        )  # size (76, 512)
        target_output = cap_emb_decoder_target  # size (76, 512)

        return {
            "decoder_input": decoder_input.to("cpu", dtype=torch.float32),
            "target_output": target_output.to("cpu", dtype=torch.float32),
            "attention_mask": attention_mask.to("cpu", dtype=torch.float32),
        }
