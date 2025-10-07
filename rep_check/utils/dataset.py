import zarr
import torch
from einops import rearrange
from typing import Optional
from torch.utils.data import Dataset
from rep_check.utils.normalizer import Normalizer


class PoseLandmarkDataset(Dataset):

    def __init__(
        self,
        zarr_path: str,
        normalizer: Optional[Normalizer] = None
    ) -> None:
        self.root = zarr.open(zarr_path, mode="r")
        self.landmarks = self.root["landmark"]
        self.labels = self.root["label"]
        if normalizer is None:
            self.normalizer = Normalizer()
            self.normalizer.fit(rearrange(self.landmarks[:], 'n c t j -> (n t) c j'))
        else:
            self.normalizer = normalizer

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmark = self.landmarks[idx]
        label = self.labels[idx]
        landmark = torch.from_numpy(landmark).float()
        label = torch.tensor(label, dtype=torch.long)
        landmark = self.normalizer.normalize(rearrange(landmark, 'c t j -> t c j'))
        landmark = rearrange(landmark, 't c j -> c t j')   
        return landmark, label
    
    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def get_normalizer(self) -> Normalizer:
        return self.normalizer