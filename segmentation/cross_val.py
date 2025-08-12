from segmentation.dataset import DRSegmentationDataset
from segmentation.unet_xception import UNet
from segmentation.dice_loss import DiceLoss
import torch
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import mlflow
from segmentation.utils import log_class_metrics, memory_stats, calculate_mask_metrics
import os


# Different batch sizes because in fine tuning discriminator is used so more VRAM is used
BATCH_SIZE = 8
EPOCHS = 2
K_FOLDS = 5

LEARNING_RATE = 0.0002 # Learning rate for generator used in every stage

# For sanity check
NUM_CLASSES = 4 # Depends on whether optic_disc is included or not

GPU = 0 # gpu index to be used
USE_MLFLOW = True # Whether to log metrics to mlflow
USE_TENSORBOARD = True # Whether to log metrics to tenorboard
LOG_NAME = 'run_name' # Name of log used for model saving, mlflow run name and tensorboard log
DATASET_PATH = ''
LOG_DIR = f'runs'

LOSS_FUNCTION = torch.nn.BCELoss # Can be changed to DiceLoss or any other loss function

device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

if USE_TENSORBOARD:
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, LOG_NAME))
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")

# Set manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

train_dataset = DRSegmentationDataset(os.path.join(DATASET_PATH, 'train_set'))
test_dataset = DRSegmentationDataset(os.path.join(DATASET_PATH, 'test_set'))
val_dataset = DRSegmentationDataset(os.path.join(DATASET_PATH, 'val_set'))

assert len(train_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
assert len(test_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
assert len(val_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"

# ======================================================================================================================
# K Fold training loop

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train(train_dataset, epochs, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True) 
    fold_results = {fold: 0 for fold in range(k_folds)}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        # Create samplers for train validation split
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Splits dataset to train and val
        train_loader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=BATCH_SIZE, sampler=val_subsampler)
        
        generator_model = UNet(in_channels=3, out_channels=NUM_CLASSES)
        generator_model.to(device)

        generator_model.apply(reset_weights)

        optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.5])

        loss = LOSS_FUNCTION()
        
        for epoch in range(epochs):
            training_epoch_loss = 0
            validation_epoch_loss = 0

            generator_model.train()
            for train_batch in tqdm(train_loader):
                optimizer.zero_grad()
                
                input_tensor = train_batch[0].to(device)
                target_tensor = train_batch[1].to(device)

                train_output = generator_model(input_tensor)

                loss_value = loss(train_output, target_tensor)
                loss_value.backward()
                optimizer.step()

                training_epoch_loss += loss_value.detach().cpu().item()

            training_epoch_loss /= len(train_loader)

            if USE_TENSORBOARD:
                writer.add_scalar(f'Fold{fold}/Loss/Train', training_epoch_loss, epoch)
            if USE_MLFLOW:
                mlflow.log_metric(f'Fold{fold}/Loss/Train', training_epoch_loss, step=epoch)

            if epoch % 5 == 0:
                all_metrics, validation_epoch_loss, val_auroc, val_auprc, val_accuracy, val_f1_score = calculate_mask_metrics(val_loader, generator_model, loss, train_dataset.class_names, device)
                log_class_metrics(all_metrics, epoch, split="cross_val", log_mlflow=USE_MLFLOW, tensorboard_writer=writer)
                
                if USE_TENSORBOARD:
                    writer.add_scalar(f'Fold{fold}/Loss/Val', validation_epoch_loss, epoch)
                    writer.add_scalar(f'Fold{fold}/AUROC/Val', val_auroc, epoch)
                    writer.add_scalar(f'Fold{fold}/AUPRC/Val', val_auprc, epoch)
                    writer.add_scalar(f'Fold{fold}/Accuracy/Val', val_accuracy, epoch)
                    writer.add_scalar(f'Fold{fold}/F1Score/Val', val_f1_score, epoch)
                if USE_MLFLOW:
                    mlflow.log_metric(f'Fold{fold}/Loss/Val', validation_epoch_loss, step=epoch)
                    mlflow.log_metric(f'Fold{fold}/AUROC/Val', val_auroc, step=epoch)
                    mlflow.log_metric(f'Fold{fold}/AUPRC/Val', val_auprc, step=epoch)
                    mlflow.log_metric(f'Fold{fold}/Accuracy/Val', val_accuracy, step=epoch)
                    mlflow.log_metric(f'Fold{fold}/F1Score/Val', val_f1_score, step=epoch)

            print(f"Fold: {fold}, Epoch: {epoch}, Mean training loss: {training_epoch_loss}")
        fold_results[fold] = validation_epoch_loss
    
    final_fold_values = [value for k, value in fold_results.items()]
    average_validation_result = np.mean(final_fold_values)
    print("Average validation result:", average_validation_result)

    writer.add_scalar("Average validation result:", average_validation_result)


def main():
    if USE_MLFLOW:
        with mlflow.start_run(run_name=f"{LOG_NAME}_cross_validation"):
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("k_folds", K_FOLDS)
            train(train_dataset, EPOCHS, k_folds=K_FOLDS)
    else:
        train(train_dataset, EPOCHS, k_folds=K_FOLDS)
        
if __name__ == "__main__":
    main()