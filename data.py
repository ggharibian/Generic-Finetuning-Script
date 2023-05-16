from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
from typing import Callable, Iterable, Optional, Sequence, Union

from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

class Custom_Dataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, transform = None, label_transform = None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        data = self.data.iloc[index]
        label = self.labels.iloc[index]
        if self.transform:
            data = self.transform(data)
        if self.label_transform:
            label = self.label_transform(label)
        return data, label

    def __len__(self):
        return len(self.data)