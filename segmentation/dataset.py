from torch.utils.data import Dataset
import os
import torch
from torchvision.io import read_image


class DRSegmentationDataset(Dataset):
    def __init__(self, input_path, use_masks=True):
        super(DRSegmentationDataset, self).__init__()

        images_path = os.path.join(input_path, "images")

        self.images_paths = [os.path.join(images_path, image_path) for image_path in sorted(os.listdir(images_path))]

        if use_masks:
            masks_path = os.path.join(input_path, "masks")
            self.masks_dirs = {mask_dir: [] for mask_dir in sorted(os.listdir(masks_path))}

            self.masks = []
            for mask_dir in self.masks_dirs.keys():
                for mask_path in sorted(os.listdir(os.path.join(masks_path, mask_dir))):
                    mask_path = os.path.join(masks_path, mask_dir, mask_path)
                    self.masks_dirs[mask_dir].append(mask_path)

        self.use_masks = use_masks

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        img = read_image(self.images_paths[index])
        img = (img/255).to(torch.float32)

        input = img

        if self.use_masks:
            masks = []
            for mask_dir in self.masks_dirs.keys():
                mask = read_image(self.masks_dirs[mask_dir][index])
                mask = (mask/255).to(torch.float32).squeeze()
                masks.append(mask)
            targets = torch.stack(masks)
        else:
            targets = input

        return input, targets