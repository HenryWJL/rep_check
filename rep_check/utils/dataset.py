import zarr
import torch
from einops import rearrange
from torch.utils.data import Dataset
from rep_check.utils.normalizer import Normalizer


class PoseLandmarkDataset(Dataset):

    def __init__(self, zarr_path: str) -> None:
        self.root = zarr.open(zarr_path, mode="r")
        self.landmarks = self.root["landmark"]
        self.labels = self.root["label"]
        self.normalizer = Normalizer()
        self.normalizer.fit(rearrange(self.landmarks[:], 'n c t j -> (n t j) c'), mode='mean_std')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmark = self.landmarks[idx]
        label = self.labels[idx]
        landmark = torch.from_numpy(landmark).float()
        label = torch.tensor(label, dtype=torch.long)
        landmark = self.normalizer.normalize(rearrange(landmark, 'c t j -> (t j) c'))
        landmark = rearrange(landmark, '(t j) c -> c t j', j=23)   
        return landmark, label
    
    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def get_normalizer(self) -> Normalizer:
        return self.normalizer