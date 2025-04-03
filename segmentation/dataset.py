from torch.utils.data import Dataset
import os
import torch
from torchvision.io import read_image


class DRSegmentationDataset(Dataset):
    def __init__(self, input_path):
        super(DRSegmentationDataset, self).__init__()

        images_path = os.path.join(input_path, "images")

        self.images = []
        for image_path in sorted(os.listdir(images_path)):
            img = read_image(os.path.join(images_path, image_path))
            img = (img/255).to(torch.float32)
            self.images.append(img)

        masks_path = os.path.join(input_path, "masks")
        self.masks_dirs = sorted(os.listdir(masks_path))

        self.masks = []
        for mask_dir in self.masks_dirs:
            masks = []
            for mask_path in sorted(os.listdir(os.path.join(masks_path, mask_dir))):
                mask = read_image(os.path.join(masks_path, mask_dir, mask_path))
                mask = (mask/255).to(torch.float32)
                masks.append(mask)
            masks = torch.stack(masks)
            self.masks.append(masks)

        self.images = torch.stack(self.images)
        self.masks = torch.stack(self.masks)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        input = self.images[index]
        targets = self.masks[:, index].squeeze()

        return input, targets