from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset, dict_transform):
        self.dataset = dataset
        self.dict_transform = dict_transform

    def __getitem__(self, item):
        data = self.dataset[item]
        if self.dict_transform is not None:
            data = self.dict_transform(data)
        return data

    def __len__(self):
        return len(self.dataset)
