from segmentation.dataset import DRSegmentationDataset
from segmentation.unet_xception import UNet
from segmentation.discriminator import Discriminator
from segmentation.dice_loss import DiceLoss
import torch
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
from tqdm import tqdm
import mlflow
from typing import List, Dict


# Different batch sizes because in fine tuning discriminator is used so more VRAM is used
PRETRAINING_BATCH_SIZE = 8
FINE_TUNING_BATCH_SIZE = 8

LEARNING_RATE = 0.0002 # Learning rate for generator used in every stage
DISCRIMINATOR_LEARNING_RATE = 1e-6 # Learning rate for discriminator used in fine tuning

NUM_CLASSES = 5 # Depends on whether optic_disc is included or not

GPU = 0 # gpu index to be used
LAMBDA = 10 
CROSS_VALIDATION = True # Whether to perform cross validation
USE_MLFLOW = False # Whether to log metrics to mlflow
USE_TENSORBOARD = True # Whether to log metrics to tenorboard
LOG_NAME = "xception_dice_loss" # Name of log used for model saving, mlflow run name and tensorboard log

device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if USE_TENSORBOARD:
    writer = SummaryWriter(log_dir=f"/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/runs/{LOG_NAME}")
if USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")

# Set manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

train_dataset = DRSegmentationDataset("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/processed_segmentation_dataset/train_set")
test_dataset = DRSegmentationDataset("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/processed_segmentation_dataset/test_set")
val_dataset = DRSegmentationDataset("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/processed_segmentation_dataset/val_set")

assert len(train_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
assert len(test_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"
assert len(val_dataset.class_names) == NUM_CLASSES, "Number of classes in dataset is not equal to defined number of classes"

image_level_train_dataset = DRSegmentationDataset("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/IDRiD/grading/train_set", use_masks=False)
image_level_test_dataset = DRSegmentationDataset("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/IDRiD/grading/test_set", use_masks=False)


def memory_stats():
    print("Memory allocated:", torch.cuda.memory_allocated()/1024**2)
    print("Memory cached:", torch.cuda.memory_reserved()/1024**2)

# ======================================================================================================================
# Functions for validation

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

def log_class_metrics(all_metrics: List[Dict], step: int, split: str="") -> None:
    for metrics in all_metrics:
        class_name = metrics['class_name']
        auroc = metrics['auroc']
        auprc = metrics['auprc']
        accuracy = metrics['accuracy']
        f1_score = metrics['f1_score']
        loss = metrics['loss']

        if USE_MLFLOW:
            mlflow.log_metric(f"{class_name}/{split}/AUROC", auroc, step=step)
            mlflow.log_metric(f"{class_name}/{split}/AUPRC", auprc, step=step)
            mlflow.log_metric(f"{class_name}/{split}/Accuracy", accuracy, step=step)
            mlflow.log_metric(f"{class_name}/{split}/F1Score", f1_score, step=step)
            mlflow.log_metric(f"{class_name}/{split}/DiceLoss", loss, step=step)

        if USE_TENSORBOARD:
            writer.add_scalar(f"{class_name}/{split}/AUROC", auroc, step)
            writer.add_scalar(f"{class_name}/{split}/AUPRC", auprc, step)
            writer.add_scalar(f"{class_name}/{split}/Accuracy", accuracy, step)
            writer.add_scalar(f"{class_name}/{split}/F1Score", f1_score, step)
            writer.add_scalar(f"{class_name}/{split}/DiceLoss", loss, step)

def calculate_mask_metrics(validation_px_dataloader, generator_model, criterion, class_names):
    targets = [[] for _ in range(NUM_CLASSES)]
    predicted_values = [[] for _ in range(NUM_CLASSES)]

    validation_loss = [0 for _ in range(NUM_CLASSES)]

    generator_model.eval()
    with torch.no_grad():
        for batch_index, px_level_batch in tqdm(enumerate(validation_px_dataloader)):
            mask_input_tensor = px_level_batch[0].to(device)
            mask_target_tensor = px_level_batch[1].to(device)

            # Generator masks for px level dataset and image level dataset
            generator_output = generator_model(mask_input_tensor)

            for i in range(NUM_CLASSES):
                loss_value = criterion(generator_output[:, i], mask_target_tensor[:, i])
                validation_loss[i] += loss_value.detach().cpu().item()

                targets[i] += torch.flatten(mask_target_tensor[:, i]).cpu().detach().tolist()
                predicted_values[i] += torch.flatten(generator_output[:, i]).cpu().detach().tolist()

            del mask_input_tensor
            del mask_target_tensor
            del generator_output
            torch.cuda.empty_cache()

    validation_loss = [class_val_loss / len(validation_px_dataloader) for class_val_loss in validation_loss]

    all_metrics = []
    for i in range(NUM_CLASSES):
        class_metrics = {
            'class_name': class_names[i],
            'loss': validation_loss[i]        
        }
        class_targets = torch.tensor(targets[i]).to(device)
        class_predicted_values = torch.tensor(predicted_values[i]).to(device)

        auroc_metric = BinaryAUROC().to(device)
        auroc_metric.update(class_predicted_values, class_targets)
        auroc = auroc_metric.compute()

        auprc_metric = BinaryAUPRC().to(device)
        auprc_metric.update(class_predicted_values, class_targets)
        auprc = auprc_metric.compute()

        accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
        accuracy_metric.update(class_predicted_values, class_targets)
        accuracy = accuracy_metric.compute()

        f1_metric = BinaryF1Score(threshold=0.5).to(device)
        f1_metric.update(class_predicted_values, class_targets)
        f1_score = f1_metric.compute()

        class_metrics['auroc'] = auroc.item()
        class_metrics['auprc'] = auprc.item()
        class_metrics['accuracy'] = accuracy.item()
        class_metrics['f1_score'] = f1_score.item()
        all_metrics.append(class_metrics)

    # Mean value of metric for all classes 
    validation_epoch_loss = np.mean(validation_loss)
    val_auroc = np.mean([metrics['auroc'] for metrics in all_metrics])
    val_auprc = np.mean([metrics['auprc'] for metrics in all_metrics])
    val_accuracy = np.mean([metrics['accuracy'] for metrics in all_metrics])
    val_f1_score = np.mean([metrics['f1_score'] for metrics in all_metrics])


    return all_metrics, validation_epoch_loss, val_auroc, val_auprc, val_accuracy, val_f1_score

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
                      batch_size=PRETRAINING_BATCH_SIZE, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=PRETRAINING_BATCH_SIZE, sampler=val_subsampler)
        
        generator_model = UNet(in_channels=3, out_channels=NUM_CLASSES)
        generator_model.to(device)

        generator_model.apply(reset_weights)

        optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.5])

        loss = DiceLoss()
        
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
                all_metrics, validation_epoch_loss, val_auroc, val_auprc, val_accuracy, val_f1_score = calculate_mask_metrics(val_loader, generator_model, loss, train_dataset.class_names)
                log_class_metrics(all_metrics, epoch, split="cross_val")
                
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

if CROSS_VALIDATION:
    epochs = 300
    k_folds = 5
    if USE_MLFLOW:
        with mlflow.start_run(run_name=f"{LOG_NAME}_cross_validation"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", PRETRAINING_BATCH_SIZE)
            mlflow.log_param("k_folds", k_folds)
            train(train_dataset, epochs, k_folds=k_folds)
    else:
        train(train_dataset, epochs, k_folds=k_folds)

# ======================================================================================================================
# Generator pretraining

px_level_train_dataloader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=PRETRAINING_BATCH_SIZE)
px_level_val_dataloader = torch.utils.data.DataLoader(
                      val_dataset, 
                      batch_size=PRETRAINING_BATCH_SIZE)
px_level_test_dataloader = torch.utils.data.DataLoader(
                      test_dataset, 
                      batch_size=1)

def train(train_dataloader, val_dataloader, epochs, generator_model):
    optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.5])

    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        training_epoch_loss = 0
        
        generator_model.train()
        for px_level_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            # Generator masks for px level dataset and image level dataset
            px_level_input_tensor = px_level_batch[0].to(device)
            px_level_target_tensor = px_level_batch[1].to(device)
            px_level_output_masks = generator_model(px_level_input_tensor)

            loss_value = criterion(px_level_output_masks, px_level_target_tensor)

            loss_value.backward()
            optimizer.step()

            training_epoch_loss += loss_value.item()

            del px_level_input_tensor
            del px_level_target_tensor
            del px_level_output_masks
            torch.cuda.empty_cache()


        all_metrics, training_epoch_loss, train_generator_auroc, train_generator_auprc, train_generator_accuracy, train_generator_f1_score = calculate_mask_metrics(train_dataloader, generator_model, criterion, train_dataset.class_names)
        log_class_metrics(all_metrics, epoch, split='train')

        all_metrics, validation_epoch_loss, val_generator_auroc, val_generator_auprc, val_generator_accuracy, val_generator_f1_score = calculate_mask_metrics(val_dataloader, generator_model, criterion, val_dataset.class_names)
        log_class_metrics(all_metrics, epoch, split='val')
        
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

generator_model = UNet(3, NUM_CLASSES)
generator_model.to(device)

if USE_MLFLOW:
    with mlflow.start_run(run_name=f'{LOG_NAME}_pretraining'):
        mlflow.log_param("Batch size", PRETRAINING_BATCH_SIZE)
        mlflow.log_param("Learning rate", LEARNING_RATE)
        train(px_level_train_dataloader, px_level_val_dataloader, 2, generator_model)
else:
    train(px_level_train_dataloader, px_level_val_dataloader, 2, generator_model)

torch.save(generator_model.state_dict(), f"{LOG_NAME}_pretrained.pth")

del generator_model
torch.cuda.empty_cache()

# ======================================================================================================================
# Final training with Discriminator

px_level_train_dataloader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=FINE_TUNING_BATCH_SIZE)

image_level_train_dataloader = torch.utils.data.DataLoader(image_level_train_dataset, batch_size=FINE_TUNING_BATCH_SIZE)
image_level_test_dataloader = torch.utils.data.DataLoader(image_level_test_dataset, batch_size=1)

def train(train_dataloader, image_level_dataloader, epochs, generator_model, discriminator_model):
        optimizer = torch.optim.Adam(generator_model.parameters(), lr=LEARNING_RATE, betas=[0.5, 0.5])
        discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)

        loss = torch.nn.BCELoss()
        discriminator_loss = torch.nn.BCELoss()
        
        for epoch in range(epochs):
            training_epoch_loss = 0

            discriminator_epoch_loss = 0
            gan_epoch_loss = 0
            mask_epoch_loss = 0
            
            for px_level_batch, img_level_batch in tqdm(zip(train_dataloader, image_level_dataloader)):
                # Discriminator training
                generator_model.eval()
                discriminator_model.train()
                discriminator_optimizer.zero_grad()
                
                px_level_input_tensor = px_level_batch[0].to(device)
                img_level_input_tensor = img_level_batch[0].to(device)

                # Generator masks for px level dataset and image level dataset
                px_level_output_masks = generator_model(px_level_input_tensor)
                img_level_output = generator_model(img_level_input_tensor)

                px_level_output = torch.concat((px_level_output_masks, px_level_input_tensor), dim=1) # Stack with original RGB image
                img_level_output = torch.concat((img_level_output, img_level_input_tensor), dim=1) # Stack with original RGB image

                px_level_target = torch.Tensor([[1] for i in range(px_level_output.shape[0])])
                img_level_target = torch.Tensor([[0] for i in range(img_level_output.shape[0])])

                discriminator_input = torch.concat((px_level_output, img_level_output), dim=0)
                discriminator_target = torch.concat((px_level_target, img_level_target), dim=0)

                indices = torch.randperm(discriminator_input.shape[0])
                discriminator_input=discriminator_input[indices, ...].to(device)
                discriminator_target=discriminator_target[indices, ...].to(device)

                discriminator_output = discriminator_model(discriminator_input.detach())

                discriminator_loss_value = discriminator_loss(discriminator_output, discriminator_target)

                discriminator_loss_value.backward()
                discriminator_optimizer.step()

                discriminator_epoch_loss += discriminator_loss_value

                # Generator Training
                generator_model.train()
                discriminator_model.eval()
                optimizer.zero_grad()

                # Generator masks for px level dataset and image level dataset
                px_level_output_masks = generator_model(px_level_input_tensor)
                img_level_output = generator_model(img_level_input_tensor)

                px_level_mask_target_tensor = px_level_batch[1].to(device)

                img_level_output = torch.concat((img_level_output, img_level_input_tensor), dim=1) # Stack with original RGB image

                discriminator_output = discriminator_model(img_level_output)
                generator_gan_loss = discriminator_loss(discriminator_output, px_level_target.to(device))
                mask_loss = loss(px_level_output_masks, px_level_mask_target_tensor)

                loss_value = generator_gan_loss + LAMBDA*mask_loss
                loss_value.backward()
                optimizer.step()

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

            discriminator_auroc, discriminator_auprc, discriminator_accuracy, discriminator_f1_score = discriminator_validation(train_dataloader, image_level_dataloader, generator_model, discriminator_model)
            mean_discriminator_loss = discriminator_epoch_loss / len(train_dataloader)
            mean_gan_epoch_loss = gan_epoch_loss / len(train_dataloader)
            mean_mask_epoch_loss = mask_epoch_loss / len(train_dataloader)

            all_metrics, generator_loss, generator_auroc, generator_auprc, generator_accuracy, generator_f1_score = calculate_mask_metrics(train_dataloader, generator_model, loss, train_dataset.class_names)
            log_class_metrics(all_metrics, epoch, split='train_fine_tune')
            training_epoch_loss /= len(train_dataloader)

            if USE_TENSORBOARD:
                writer.add_scalar('Final/GeneratorLoss/Train', training_epoch_loss, epoch)
                writer.add_scalar('Final/GeneratorGANLoss/Train', mean_gan_epoch_loss, epoch)
                writer.add_scalar('Final/GeneratorMaskLoss/Train', mean_mask_epoch_loss, epoch)
                writer.add_scalar('Final/GeneratorAUROC', generator_auroc, epoch)
                writer.add_scalar('Final/GeneratorAUPRC', generator_auprc, epoch)
                writer.add_scalar('Final/GeneratorAccuracy', generator_accuracy, epoch)
                writer.add_scalar('Final/GeneratorF1Score', generator_f1_score, epoch)

                writer.add_scalar('Final/DiscriminatorLoss/Train', mean_discriminator_loss, epoch)
                writer.add_scalar('Final/DiscriminatorAUROC', discriminator_auroc, epoch)
                writer.add_scalar('Final/DiscriminatorAUPRC', discriminator_auprc, epoch)
                writer.add_scalar('Final/DiscriminatorAccuracy', discriminator_accuracy, epoch)
                writer.add_scalar('Final/DiscriminatorF1Score', discriminator_f1_score, epoch)
            if USE_MLFLOW:
                mlflow.log_metric('Final/GeneratorLoss/Train', training_epoch_loss, step=epoch)
                mlflow.log_metric('Final/GeneratorGANLoss/Train', mean_gan_epoch_loss, step=epoch)
                mlflow.log_metric('Final/GeneratorMaskLoss/Train', mean_mask_epoch_loss, step=epoch)
                mlflow.log_metric('Final/GeneratorAUROC', generator_auroc, step=epoch)
                mlflow.log_metric('Final/GeneratorAUPRC', generator_auprc, step=epoch)
                mlflow.log_metric('Final/GeneratorAccuracy', generator_accuracy, step=epoch)
                mlflow.log_metric('Final/GeneratorF1Score', generator_f1_score, step=epoch)

                mlflow.log_metric('Final/DiscriminatorLoss/Train', mean_discriminator_loss, step=epoch)
                mlflow.log_metric('Final/DiscriminatorAUROC', discriminator_auroc, step=epoch)
                mlflow.log_metric('Final/DiscriminatorAUPRC', discriminator_auprc, step=epoch)
                mlflow.log_metric('Final/DiscriminatorAccuracy', discriminator_accuracy, step=epoch)
                mlflow.log_metric('Final/DiscriminatorF1Score', discriminator_f1_score, step=epoch)
            print(f"Epoch: {epoch}, Mean training loss: {training_epoch_loss}, Mean discriminator loss: {mean_discriminator_loss}, Discriminator accuracy: {discriminator_accuracy}")
            
            memory_stats()

generator_model = UNet(3, NUM_CLASSES)
generator_model.load_state_dict(torch.load(f"{LOG_NAME}_pretrained.pth"))
generator_model.to(device)

discriminator_model = Discriminator()
discriminator_model.to(device)

if USE_MLFLOW:
    with mlflow.start_run(run_name=f'{LOG_NAME}_fine_tuning'):
        mlflow.log_param("Batch size", FINE_TUNING_BATCH_SIZE)
        mlflow.log_param("Learning rate", LEARNING_RATE)
        mlflow.log_param("Discriminator learning rate", DISCRIMINATOR_LEARNING_RATE)
        mlflow.log_param("Lambda", LAMBDA)
        train(px_level_train_dataloader, image_level_train_dataloader, 10, generator_model, discriminator_model=discriminator_model)
else:
    train(px_level_train_dataloader, image_level_train_dataloader, 10, generator_model, discriminator_model=discriminator_model)
torch.save(generator_model.state_dict(), f"{LOG_NAME}_fine_tuned.pth")

del generator_model
torch.cuda.empty_cache()

# ======================================================================================================================
# Test final model

generator_model = UNet(3, NUM_CLASSES)
generator_model.load_state_dict(torch.load(f"{LOG_NAME}_fine_tuned.pth", map_location=device))
generator_model.to(device)

generator_model.eval()
loss = torch.nn.BCELoss()
test_loss = 0

with torch.no_grad():
    for test_batch_id, test_batch in enumerate(px_level_test_dataloader):                
        input_tensor = test_batch[0].to(device)
        target_tensor = test_batch[1].to(device)

        val_output = generator_model(input_tensor)

        loss_value = loss(val_output, target_tensor)
        test_loss += loss_value.item() 

mean_test_loss = test_loss / len(px_level_test_dataloader)
print("Mean test loss:", mean_test_loss)

for test_batch_id, test_batch in enumerate(px_level_test_dataloader):                
    input_tensor = test_batch[0].to(device)
    target_tensor = test_batch[1].to(device)

    test_output = generator_model(input_tensor)

    target_tensor = target_tensor.squeeze()
    test_output = test_output.squeeze()

    lesion_index = 4

    fig = plt.figure()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=2)

    ax = fig.add_subplot(2,1,1)
    ax.imshow(target_tensor[lesion_index, ...].cpu(), cmap='gray')
    ax.set_title("Ground truth")

    ax = fig.add_subplot(2,1, 2)
    ax.imshow(test_output[lesion_index, ...].cpu().detach(), cmap='gray')
    ax.set_title("Predicted")