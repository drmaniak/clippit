import os
from pathlib import Path

import torch
from datasets import Image, load_dataset
from torch.utils.data import DataLoader, Dataset


class Flicker30K(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
