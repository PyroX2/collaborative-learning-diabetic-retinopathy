from src.segmentation.dataset import DRSegmentationDataset
from src.segmentation.unet import UNet
from src.segmentation.discriminator import Discriminator
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
from tqdm import tqdm
import mlflow
import os
from src.segmentation.utils import log_class_metrics, calculate_mask_metrics
import yaml


# Set manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 2

LEARNING_RATE = 0.0002 # Learning rate for generator used in every stage
DISCRIMINATOR_LEARNING_RATE = 1e-6 # Learning rate for discriminator used in fine tuning

NUM_CLASSES = 5 # Depends on whether optic_disc is included or not

GPU = 0 # gpu index to be used
USE_MLFLOW = True # Whether to log metrics to mlflow
USE_TENSORBOARD = True # Whether to log metrics to tenorboard
LOG_NAME = '' # Name of log used for model saving, mlflow run name and tensorboard log
PX_DATASET_DIR = ''
IMG_DATASET_DIR = ''
CHECKPOINT_DIR = ''

with open("config/collaborative_config.yaml") as f:
    params = yaml.safe_load(f)

LAMBDA = params["lambda"]

device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if USE_TENSORBOARD:
    writer = SummaryWriter(log_dir=f"runs/{LOG_NAME}")
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")

# Function for measuring discriminator performance
def discriminator_validation(validation_px_dataloader, validation_image_level_dataloader, generator_model, discriminator_model):
    validation_accuracy = []

    targets = []
    predicted_values = []

    generator_model.eval()
    discriminator_model.eval()
    for px_level_batch, img_level_batch in tqdm(zip(validation_px_dataloader, validation_image_level_dataloader)):
        
        px_level_input_tensor = px_level_batch[0].to(device)
        img_level_input_tensor = img_level_batch[0].to(device)

        # Generator masks for px level dataset and image level dataset
        px_level_output = generator_model(px_level_input_tensor)
        img_level_output = generator_model(img_level_input_tensor)

        px_level_output = torch.concat((px_level_output, px_level_input_tensor), dim=1) # Stack with original RGB image
        img_level_output = torch.concat((img_level_output, img_level_input_tensor), dim=1) # Stack with original RGB image

        px_level_target = torch.Tensor([[1] for i in range(px_level_output.shape[0])])
        img_level_target = torch.Tensor([[0] for i in range(img_level_output.shape[0])])

        discriminator_input = torch.concat((px_level_output, img_level_output), dim=0)
        discriminator_target = torch.concat((px_level_target, img_level_target), dim=0)

        indices = torch.randperm(discriminator_input.shape[0])
        discriminator_input = discriminator_input[indices, ...].to(device)
        discriminator_target = discriminator_target[indices, ...].to(device)

        discriminator_output = discriminator_model(discriminator_input)

        discriminator_output = torch.where(discriminator_output > 0.5, 1, 0)

        batch_sum = torch.sum(discriminator_output == discriminator_target) 
        batch_accuracy = batch_sum / discriminator_output.size()[0]
        validation_accuracy.append(batch_accuracy)

        targets += discriminator_target.squeeze().cpu().detach().tolist()
        predicted_values += discriminator_output.squeeze().cpu().detach().tolist()

    targets = torch.tensor(targets).to(device)
    predicted_values = torch.tensor(predicted_values).to(device)

    auroc_metric = BinaryAUROC().to(device)
    auroc_metric.update(predicted_values, targets)
    auroc = auroc_metric.compute()

    auprc_metric = BinaryAUPRC().to(device)
    auprc_metric.update(predicted_values, targets)
    auprc = auprc_metric.compute()

    accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
    accuracy_metric.update(predicted_values, targets)
    accuracy = accuracy_metric.compute()

    f1_metric = BinaryF1Score(threshold=0.5).to(device)
    f1_metric.update(predicted_values, targets)
    f1_score = f1_metric.compute()

    return auroc, auprc, accuracy, f1_score


def train(models, dataloaders, optimizers, criterions):
    segmentation_model = models["unet"]
    discriminator_model = models["discriminator"]

    unet_optimizer = optimizers["unet"]
    discriminator_optimizer = optimizers["discriminator"]

    unet_loss = criterions["unet"]
    discriminator_loss = criterions["discriminator"]

    px_level_train_dataloader = dataloaders["px_train"]
    px_level_val_dataloader = dataloaders["px_val"]

    image_level_train_dataloader = dataloaders["img_train"]
    image_level_val_dataloader = dataloaders["img_val"]

    training_epoch_loss = 0

    discriminator_epoch_loss = 0
    gan_epoch_loss = 0
    mask_epoch_loss = 0
    
    for px_level_batch, img_level_batch in tqdm(zip(px_level_train_dataloader, image_level_train_dataloader)):
        # ========================================
        # Discriminator training

        # Set models mode
        segmentation_model.eval()
        discriminator_model.train()

        # Zero gradinet
        discriminator_optimizer.zero_grad()
        
        px_level_input_tensor = px_level_batch[0].to(device)
        img_level_input_tensor = img_level_batch[0].to(device)

        # Generator masks for px level dataset and image level dataset
        px_level_output_masks = segmentation_model(px_level_input_tensor)
        img_level_output = segmentation_model(img_level_input_tensor)

        px_level_output = torch.concat((px_level_output_masks, px_level_input_tensor), dim=1) # Stack with original RGB image
        img_level_output = torch.concat((img_level_output, img_level_input_tensor), dim=1) # Stack with original RGB image

        # Create target tensors
        px_level_target = torch.Tensor([[1] for i in range(px_level_output.shape[0])])
        img_level_target = torch.Tensor([[0] for i in range(img_level_output.shape[0])])

        # Concatenate image and pixel level inputs and targets
        discriminator_input = torch.concat((px_level_output, img_level_output), dim=0)
        discriminator_target = torch.concat((px_level_target, img_level_target), dim=0)

        # Shuffle
        indices = torch.randperm(discriminator_input.shape[0])
        discriminator_input=discriminator_input[indices, ...].to(device)
        discriminator_target=discriminator_target[indices, ...].to(device)

        # Get discriminator prediction
        discriminator_output = discriminator_model(discriminator_input.detach())

        # Train discriminator
        discriminator_loss_value = discriminator_loss(discriminator_output, discriminator_target)
        discriminator_loss_value.backward()
        discriminator_optimizer.step()

        discriminator_epoch_loss += discriminator_loss_value

        # ========================================
        # Generator Training

        # Set models mode
        segmentation_model.train()
        discriminator_model.eval()
        unet_optimizer.zero_grad()

        # Generator masks for px level dataset and image level dataset
        px_level_output_masks = segmentation_model(px_level_input_tensor)
        img_level_output = segmentation_model(img_level_input_tensor)

        px_level_mask_target_tensor = px_level_batch[1].to(device)

        img_level_output = torch.concat((img_level_output, img_level_input_tensor), dim=1) # Stack with original RGB image

        # Create new "fake" target tensor
        px_level_target = torch.Tensor([[1] for i in range(img_level_input_tensor.shape[0])])

        discriminator_output = discriminator_model(img_level_output)
        generator_gan_loss = discriminator_loss(discriminator_output, px_level_target.to(device))
        mask_loss = unet_loss(px_level_output_masks, px_level_mask_target_tensor)

        loss_value = generator_gan_loss + LAMBDA*mask_loss
        loss_value.backward()
        unet_optimizer.step()

        training_epoch_loss += loss_value.item()
        gan_epoch_loss += generator_gan_loss.item()
        mask_epoch_loss += mask_loss.item()

        del discriminator_output
        del px_level_output_masks
        del img_level_output
        del discriminator_input
        del discriminator_target
        del px_level_input_tensor
        del px_level_target
        del img_level_input_tensor
        torch.cuda.empty_cache()

    training_epoch_loss /= len(px_level_train_dataloader)
    mean_gan_epoch_loss = gan_epoch_loss / len(px_level_train_dataloader)
    mean_mask_epoch_loss = mask_epoch_loss / len(px_level_train_dataloader)
    mean_discriminator_loss = discriminator_epoch_loss / len(px_level_train_dataloader)

    discriminator_auroc, discriminator_auprc, discriminator_accuracy, discriminator_f1_score = discriminator_validation(px_level_val_dataloader, image_level_val_dataloader, segmentation_model, discriminator_model)
    all_metrics, val_generator_loss, generator_auroc, generator_auprc, generator_accuracy, generator_f1_score = calculate_mask_metrics(px_level_val_dataloader, segmentation_model, unet_loss, ["haemmorrhages", "hard_exudates", "microaneurysms", "optic_disc", "soft_exudates"], device)

    metrics = {
        'GeneratorLoss/Train': training_epoch_loss,
        'GeneratorGANLoss/Train': mean_gan_epoch_loss,
        'GeneratorMaskLoss/Train': mean_mask_epoch_loss,
        'GeneratorLoss/Val': val_generator_loss,
        'GeneratorAUROC/Val': generator_auroc,
        'GeneratorAUPRC/Val': generator_auprc,
        'GeneratorAccuracy/Val': generator_accuracy,
        'GeneratorF1Score/Val': generator_f1_score,
        'DiscriminatorLoss/Train': mean_discriminator_loss,
        'DiscriminatorAUROC/Val': discriminator_auroc,
        'DiscriminatorAUPRC/Val': discriminator_auprc,
        'DiscriminatorAccuracy/Val': discriminator_accuracy,
        'DiscriminatorF1Score/Val': discriminator_f1_score
    }
    return metrics


def save_attention_maps():
    pass
    # reference_image = input_batch[0].cpu().detach().numpy()
    # reference_image = reference_image.transpose(1, 2, 0)

    # # TODO: Add min max scaling
    # reference_image = (reference_image * 255).astype('uint8')
    # if not os.path.exists("output_masks"):
    #     os.makedirs("output_masks")
    # cv2.imwrite("output_masks/reference_image.png", reference_image)
    
    # for i in range(NUM_LESIONS):
    #     attention_map = attention_maps[0, i].cpu().detach().numpy()
    #     print(f"{attention_map.min()}, {attention_map.max()}")
    #     attention_map = (attention_map * 255).astype('uint8')

    #     mask = masks[0][i].cpu().detach().numpy()
    #     mask = (mask*255).astype('uint8')

    #     output_path = f"output_masks/mask_{i}/"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     cv2.imwrite(os.path.join(output_path, f"mask_{epoch}.png"), attention_map)
    #     cv2.imwrite(os.path.join(output_path, f"mask_generator.png"), mask)

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    px_level_train_dataset = DRSegmentationDataset(os.path.join(PX_DATASET_DIR, 'train_set'))
    px_level_val_dataset = DRSegmentationDataset(os.path.join(PX_DATASET_DIR, 'val_set'))
    image_level_train_dataset = DRSegmentationDataset(os.path.join(IMG_DATASET_DIR, 'train_set'), use_masks=False)
    image_level_val_dataset = DRSegmentationDataset(os.path.join(IMG_DATASET_DIR, 'val_set'), use_masks=False)

    assert len(px_level_train_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
    assert len(px_level_val_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"

    generator_model = UNet(3, NUM_CLASSES)
    generator_model.load_state_dict(torch.load("segmentation_generator_pretrained.pth"))
    generator_model.to(device)

    discriminator_model = Discriminator()
    discriminator_model.to(device)

    if USE_MLFLOW:
        with mlflow.start_run(run_name=f'{LOG_NAME}_fine_tuning'):
            mlflow.log_param("Batch size", BATCH_SIZE)
            mlflow.log_param("Learning rate", LEARNING_RATE)
            mlflow.log_param("Discriminator learning rate", DISCRIMINATOR_LEARNING_RATE)
            mlflow.log_param("Lambda", LAMBDA)
            train(px_level_train_dataset, image_level_train_dataset, px_level_val_dataset, image_level_val_dataset, 10, generator_model, discriminator_model=discriminator_model)
    else:
        train(px_level_train_dataset, image_level_train_dataset, px_level_val_dataset, image_level_val_dataset, 10, generator_model, discriminator_model=discriminator_model)

if __name__ == "__main__":
    main()