import os
import cv2
import numpy as np


def merge_masks(root_dir: str) -> None:
    masks_dir_path = os.path.join(root_dir, "masks")
    images_dir_path = os.path.join(root_dir, "images")
    output_path = os.path.join(root_dir, "merged_masks")

    if not os.path.exists(masks_dir_path) or not os.path.exists(images_dir_path):
        raise ValueError("Input masks or images directory doesn't exist")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    masks_paths = []

    for mask_class in os.listdir(masks_dir_path):
        if mask_class == ".DS_Store":
            continue

        class_path = os.path.join(masks_dir_path, mask_class)
        paths = [os.path.join(class_path, mask_filename) for mask_filename in sorted(os.listdir(class_path))]
        masks_paths.append(paths)

    image_filenames = [filename for filename in sorted(os.listdir(images_dir_path)) if filename != ".DS_Store"]

    for i, cur_masks_paths in enumerate(zip(*masks_paths)):
        masks = []
        for mask_path in cur_masks_paths:
            mask = cv2.imread(mask_path, 0)
            masks.append(mask)

        classes = np.arange(1, len(masks)+1)

        masks = np.array(masks) / 255
        masks_combined = np.sum(masks, axis=0)
        
        if masks_combined.max() > 1:
            raise ValueError("Masks for each class cannot overlap")
        
        classes = classes[..., None, None]
        new_mask = np.multiply(masks, classes).sum(axis=0).astype(np.uint8)

        output_filename = image_filenames[i]
        output_image_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_image_path, new_mask)


def main():
    root_dirs = ["datasets/processed_segmentation_dataset/val_set",
                 "datasets/processed_segmentation_dataset/train_set",
                 "datasets/processed_segmentation_dataset/test_set"]
    
    for root_dir in root_dirs:
        merge_masks(root_dir)


if __name__ == "__main__":
    main()