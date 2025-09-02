from src.grading_model.grading_model import GradingModel
from src.segmentation.dataset import DRSegmentationDataset
from src.segmentation.unet import UNet
from src.segmentation.discriminator import Discriminator
from src.segmentation.dice_loss import DiceLoss
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Dict
from src.collaborative.grading_utils import train as grading_train
from src.collaborative.seg_utils import train as segmentation_train
import mlflow
import yaml


# Set manual seed for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# General
MLFLOW = True
TENSORBOARD = True
CHECKPOINT_DIR = 'models/checkpoints/collaborative/test'
IMG_LEVEL_DATASET_PATH = 'datasets/eyepacs-aptos-messidor-diabetic-retinopathy-original-preprocessed-color-enhancement'
PX_LEVEL_DATASET_PATH = 'datasets/processed_segmentation_dataset'
LOG_NAME = "attentive_grading_model_train-end_to_end"

# Segmentation model specific
NUM_SEG_CLASSES = 5 # Depends on whether optic_disc is included or not
SEG_MODEL_STATE_DICT = 'models/segmentation/segmentation_generator.pth'
UNET_OPTIMIZER_STATE_DICT = 'models/checkpoints/segmentation/segmentation_optimizer.pth'

# Grading model specific
GRADING_UPDATE_SIZE = 128 # How many samples to use in one epoch of grading model training. This value is used to balance training segmentation model using px-level data and img-level data
NUM_LESIONS = 4
NUM_GRADING_OUTPUTS = 1 # Number of outputs in grading model. 1 when used for binary classification
GRADING_MODEL_STATE_DICT = 'models/classification/grading_model_pretrain.pth'
GRADING_OPTIMIZER_STATE_DICT = 'models/classification/grading_model_pretrain_optimizer.pth'

# Load parameters from config file
with open("config/collaborative_config.yaml") as f:
    params = yaml.safe_load(f)

LAMBDA = params["lambda"]
BATCH_SIZE = params["batch_size"]
NUM_EPOCHS = params["num_epochs"]
UNET_LEARNING_RATE = params["segmentation_model_lr"]
DISCRIMINATOR_LEARNING_RATE = params["discriminator_lr"]
GRADING_LEARNING_RATE = params["grading_model_lr"]

print(type(BATCH_SIZE))
print(type(NUM_EPOCHS))
print(type(UNET_LEARNING_RATE))
print(type(GRADING_LEARNING_RATE))

if params["segmentation_loss"].lower() == "bce":
    UNET_LOSS_FUNCTION = torch.nn.BCELoss
elif params["segmentation_loss"].lower() == "dice":
    UNET_LOSS_FUNCTION = DiceLoss
else:
    raise ValueError(f"Loss function {params['segmentation_loss']} can\'t be used")


device = "cuda" if torch.cuda.is_available() else "cpu"

if TENSORBOARD:
    writer = SummaryWriter(f"runs/{LOG_NAME}")

def train(models: Dict, dataloaders: Dict, optimizers: Dict, criterions: Dict) -> None:
    best_grading_val_loss = float("inf")
    best_seg_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        metrics = grading_train(models, dataloaders, optimizers, criterions, GRADING_UPDATE_SIZE)

        grading_val_loss = metrics["Loss/validation"]

        if grading_val_loss < best_grading_val_loss:
            torch.save(models["grading"], os.path.join(CHECKPOINT_DIR, "grading_model_best_grading.pth"))
            torch.save(models["unet"], os.path.join(CHECKPOINT_DIR, "seg_model_best_grading.pth"))

        for metric_name, metric in metrics.items():
            if TENSORBOARD:
                writer.add_scalar(metric_name, metric, epoch)
            if MLFLOW:
                mlflow.log_metric(metric_name, metric, epoch)

        seg_metrics = segmentation_train(models, dataloaders, optimizers, criterions)

        seg_val_loss = seg_metrics["GeneratorLoss/Val"]

        if seg_val_loss < best_seg_loss:
            torch.save(models["unet"], os.path.join(CHECKPOINT_DIR, "seg_model_best_seg.pth"))

        for metric_name, metric in seg_metrics.items():
            if TENSORBOARD:
                writer.add_scalar(metric_name, metric, epoch)
            if MLFLOW:
                mlflow.log_metric(metric_name, metric, epoch)

    for model_name, model in models.items():
        torch.save(model, os.path.join(CHECKPOINT_DIR, f"{model_name}_last.pth"))


def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Resize((640, 640)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(15)])

    test_images_transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Resize((640, 640))])

    # Image level dataset preparation
    img_level_train_root = os.path.join(IMG_LEVEL_DATASET_PATH, "train", "two_classes")
    img_level_validation_root = os.path.join(IMG_LEVEL_DATASET_PATH, "val", "two_classes")

    img_level_train_dataset = ImageFolder(img_level_train_root, transform=transform)
    img_level_validation_dataset = ImageFolder(img_level_validation_root, transform=test_images_transform)

    _, img_level_train_metrics_dataset = random_split(img_level_train_dataset, [0.9, 0.1])

    img_level_train_dataloader = DataLoader(img_level_train_dataset, BATCH_SIZE, shuffle=True, num_workers=42)
    img_level_validation_dataloader = DataLoader(img_level_validation_dataset, BATCH_SIZE, shuffle=False, num_workers=42)
    img_level_train_metrics_dataloader = DataLoader(img_level_train_metrics_dataset, BATCH_SIZE, shuffle=False, num_workers=42)

    # Pixel level dataset preparation
    px_level_train_dataset = DRSegmentationDataset(os.path.join(PX_LEVEL_DATASET_PATH, 'train'))
    px_level_val_dataset = DRSegmentationDataset(os.path.join(PX_LEVEL_DATASET_PATH, 'val'))
    px_level_train_dataloader = torch.utils.data.DataLoader(
                        px_level_train_dataset, 
                        batch_size=BATCH_SIZE)
    px_level_val_dataloader = torch.utils.data.DataLoader(
                        px_level_val_dataset, 
                        batch_size=BATCH_SIZE)
    
    # Keep all dataloaders in dict for cleaner train function call
    dataloaders = {
        "img_train": img_level_train_dataloader,
        "img_val": img_level_validation_dataloader,
        "img_train_metrics": img_level_train_metrics_dataloader,
        "px_train": px_level_train_dataloader,
        "px_val": px_level_val_dataloader
        }

    # Initialize models
    grading_model = GradingModel(num_lesions=NUM_LESIONS, num_outputs=NUM_GRADING_OUTPUTS)
    grading_model.to(device)
    grading_model.load_state_dict(torch.load(GRADING_MODEL_STATE_DICT, weights_only=True, map_location=device))

    segmentation_model = UNet(3, NUM_SEG_CLASSES)
    segmentation_model.to(device)
    segmentation_model.load_state_dict(torch.load(SEG_MODEL_STATE_DICT, weights_only=True, map_location=device))

    discriminator_model = Discriminator()
    discriminator_model.to(device)

    models = {
        "grading": grading_model, 
        "unet": segmentation_model, 
        "discriminator": discriminator_model
        }

    # Optimizer initialization
    grading_optimizer = torch.optim.Adam(grading_model.parameters(), lr=GRADING_LEARNING_RATE)
    grading_optimizer.load_state_dict(torch.load(GRADING_OPTIMIZER_STATE_DICT, map_location=device))
    segmentation_optimizer = torch.optim.Adam(grading_model.parameters(), lr=UNET_LEARNING_RATE, betas=[0.5, 0.5])
    # segmentation_optimizer.load_state_dict(torch.load(UNET_OPTIMIZER_STATE_DICT, map_location=device))
    discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)

    optimizers = {
        "grading": grading_optimizer, 
        "unet": segmentation_optimizer, 
        "discriminator": discriminator_optimizer
        }

    # Initialize criterions
    grading_criterion = torch.nn.BCELoss()
    unet_criterion = UNET_LOSS_FUNCTION()
    discriminator_criterion = torch.nn.BCELoss()

    criterions = {
        "grading": grading_criterion, 
        "unet": unet_criterion, 
        "discriminator": discriminator_criterion
        }
    
    if MLFLOW:
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run():
            train(models, dataloaders, optimizers, criterions)
    else:
        train(models, dataloaders, optimizers, criterions)

if __name__ == '__main__':
    main()