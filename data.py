from torch.utils.data import Dataset, DataLoader
import pandas as pd

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