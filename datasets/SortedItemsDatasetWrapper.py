import torch.utils.data as data


class SortedItemsDatasetWrapper(data.Dataset):
    def __init__(self, dataset, items):
        self.dataset = dataset
        self.items = items

    def __getitem__(self, ind):
        values = self.dataset[ind]
        sorted_values = [values[item] for item in self.items]
        return sorted_values

    def __len__(self):
        return len(self.dataset)
