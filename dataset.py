import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Tuple, Any

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label 