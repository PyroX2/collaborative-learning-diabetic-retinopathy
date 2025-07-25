import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2



transforms = v2.Compose([
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

class GradingDataset(Dataset):
    def __init__(self, images_dir: str, csv_labels: str, transform=None):
        super().__init__()

        self.images_paths = sorted([os.path.join(images_dir, path) for path in os.listdir(images_dir)])
        self.labels = pd.read_csv(csv_labels).iloc[:, 1].tolist() # First column is DR grade

        self.transform = transform

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        img_path = self.images_paths[index]
        image = read_image(img_path)
        image = (image/255).to(torch.float32)

        image = transforms(image)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index] 

        return image, label
