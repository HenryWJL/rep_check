import zarr
import torch
from torch.utils.data import Dataset


class PoseLandmarkDataset(Dataset):

    def __init__(self, zarr_path: str) -> None:
        self.root = zarr.open(zarr_path, mode="r")
        self.landmarks = self.root["landmark"]
        self.labels = self.root["label"]

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmark = self.landmarks[idx]
        label = self.labels[idx]
        landmark = torch.from_numpy(landmark).float()
        label = torch.tensor(label, dtype=torch.long)        
        return landmark, label