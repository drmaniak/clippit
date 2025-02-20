from pathlib import Path

import pandas as pd
import torch
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
        img_emb, cap, img_id = self.dataset.iloc[idx][
            [
                "img_embedding",
                "caption_text",
                "img_id",
            ]
        ]

        img_emb = torch.tensor(img_emb)

        return {
            "image_emb": img_emb.to("cpu", dtype=torch.float32),
            "caption": cap,
            "img_id": img_id,
        }
