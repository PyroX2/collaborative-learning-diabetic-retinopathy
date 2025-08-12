from segmentation.dataset import DRSegmentationDataset
from segmentation.unet_xception import UNet
from segmentation.dice_loss import DiceLoss
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import mlflow
from segmentation.utils import calculate_mask_metrics, log_class_metrics
import os


# Different batch sizes because in fine tuning discriminator is used so more VRAM is used
BATCH_SIZE = 8
EPOCHS = 350

LEARNING_RATE = 0.0002 # Learning rate for generator used in every stage

NUM_CLASSES = 4 # Depends on whether optic_disc is included or not

GPU = 0 # gpu index to be used
USE_MLFLOW = False # Whether to log metrics to mlflow
USE_TENSORBOARD = True # Whether to log metrics to tenorboard

# Name of log used for model saving, mlflow run name and tensorboard log
LOG_NAME = 'test_run'

DATASET_DIR = ''

# Directory where checkpoints are saved
CHECKPOINT_DIR = ''

LOSS_FUNCTION = torch.nn.BCELoss

device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if USE_TENSORBOARD:
    writer = SummaryWriter(log_dir=f"runs/{LOG_NAME}")
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")

# Set manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(train_dataset, val_dataset, epochs, generator_model):
    optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.5])

    criterion = LOSS_FUNCTION()

    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=BATCH_SIZE)
    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=BATCH_SIZE)
    
    for epoch in range(epochs):
        training_epoch_loss = 0
        
        generator_model.train()
        for train_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            # Generator masks for px level dataset and image level dataset
            px_level_input_tensor = train_batch[0].to(device)
            px_level_target_tensor = train_batch[1].to(device)
            px_level_output_masks = generator_model(px_level_input_tensor)

            loss_value = criterion(px_level_output_masks, px_level_target_tensor)

            loss_value.backward()
            optimizer.step()

            training_epoch_loss += loss_value.item()

            del px_level_input_tensor
            del px_level_target_tensor
            del px_level_output_masks
            torch.cuda.empty_cache()

        # Get train dataset metrics and log them (with class separation)
        all_metrics, training_epoch_loss, train_generator_auroc, train_generator_auprc, train_generator_accuracy, train_generator_f1_score = calculate_mask_metrics(train_dataloader, generator_model, criterion, train_dataset.class_names, device)
        log_class_metrics(all_metrics, epoch, split='train', log_mlflow=USE_MLFLOW, tensorboard_writer=writer)

        # Get validation dataset metrics and log them (with class separation)
        all_metrics, validation_epoch_loss, val_generator_auroc, val_generator_auprc, val_generator_accuracy, val_generator_f1_score = calculate_mask_metrics(val_dataloader, generator_model, criterion, val_dataset.class_names, device)
        log_class_metrics(all_metrics, epoch, split='val', log_mlflow=USE_MLFLOW, tensorboard_writer=writer)
        
        # Logging general metrics
        if USE_TENSORBOARD:
            writer.add_scalar('GeneratorPretraining/GeneratorLoss/Train', training_epoch_loss, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAUROC/Train', train_generator_auroc, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAUPRC/Train', train_generator_auprc, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAccuracy/Train', train_generator_accuracy, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorF1Score/Train', train_generator_f1_score, epoch)

            writer.add_scalar('GeneratorPretraining/GeneratorLoss/Val', validation_epoch_loss, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAUROC/Val', val_generator_auroc, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAUPRC/Val', val_generator_auprc, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorAccuracy/Val', val_generator_accuracy, epoch)
            writer.add_scalar('GeneratorPretraining/GeneratorF1Score/Val', val_generator_f1_score, epoch)
        if USE_MLFLOW:
            mlflow.log_metric('GeneratorPretraining/GeneratorLoss/Train', training_epoch_loss, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAUROC/Train', train_generator_auroc, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAUPRC/Train', train_generator_auprc, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAccuracy/Train', train_generator_accuracy, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorF1Score/Train', train_generator_f1_score, step=epoch)

            mlflow.log_metric('GeneratorPretraining/GeneratorLoss/Val', validation_epoch_loss, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAUROC/Val', val_generator_auroc, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAUPRC/Val', val_generator_auprc, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorAccuracy/Val', val_generator_accuracy, step=epoch)
            mlflow.log_metric('GeneratorPretraining/GeneratorF1Score/Val', val_generator_f1_score, step=epoch)

        print(f"Epoch: {epoch}, Mean training loss: {training_epoch_loss}, Mean validation loss: {validation_epoch_loss}")

        if epoch % 20 == 0:
            # Every 20 epochs save model and optimizer state checkpoint
            torch.save(generator_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{LOG_NAME}_ckpt_{epoch}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, f"{LOG_NAME}_adam_ckpt_{epoch}.pth"))

    # Save final model and optimizer state
    torch.save(generator_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{LOG_NAME}_final.pth"))
    torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, f"{LOG_NAME}_adam_final.pth"))

def main():
    train_dataset = DRSegmentationDataset(os.path.join(DATASET_DIR, 'train_set'))
    val_dataset = DRSegmentationDataset(os.path.join(DATASET_DIR, 'val_set'))

    assert len(train_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
    assert len(val_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"

    generator_model = UNet(3, NUM_CLASSES)
    generator_model.to(device)

    if USE_MLFLOW:
        with mlflow.start_run(run_name=f'{LOG_NAME}_pretraining'):
            mlflow.log_param("Batch size", BATCH_SIZE)
            mlflow.log_param("Learning rate", LEARNING_RATE)
            train(train_dataset, val_dataset, EPOCHS, generator_model)
    else:
        train(train_dataset, val_dataset, EPOCHS, generator_model)

if __name__ == "__main__":
    main()