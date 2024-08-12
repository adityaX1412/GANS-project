from dataloader import CelebADataset

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

dataset_name = "tpremoli/CelebA-attrs"
dataset = load_dataset(dataset_name)
class CelebADataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if self.transform:
            image = self.transform(image)
        return image


    


