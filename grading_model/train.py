from grading_model.dataset import GradingDataset
from grading_model.grading_model import GradingModel
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from segmentation.unet import UNet

import mlflow

torch.manual_seed(0)

import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

for path in sys.path:
    print(path)


BATCH_SIZE = 16
MLFLOW = False
TENSORBOARD = True
LOG_NAME = "attentive_grading_model_train"

if MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = v2.Compose([
    v2.ToTensor(), 
    v2.Resize((640, 640)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15)])

test_images_transform = v2.Compose([
    v2.ToTensor(), 
    v2.Resize((640, 640))])

# Kaggle Joined dataset
train_root = "/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/eyepacs-aptos-messidor-diabetic-retinopathy-preprocessed/train"
validation_root = "/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/eyepacs-aptos-messidor-diabetic-retinopathy-preprocessed/val"
test_root = "/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/eyepacs-aptos-messidor-diabetic-retinopathy-preprocessed/test"

train_dataset = ImageFolder(train_root, transform=transform)
validation_dataset = ImageFolder(validation_root, transform=test_images_transform)
test_dataset = ImageFolder(test_root, transform=test_images_transform)

_, train_metrics_dataset = random_split(train_dataset, [0.9, 0.1])

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=42)
validation_dataloader = DataLoader(validation_dataset, BATCH_SIZE, shuffle=False, num_workers=42)
test_dataset = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=42)
train_metrics_dataloader = DataLoader(train_metrics_dataset, BATCH_SIZE, shuffle=False, num_workers=42)

grading_model_pretrained = GradingModel()
grading_model_pretrained.to(device)
grading_model_pretrained.load_state_dict(torch.load("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/models/classification/grading_model_pretrained.pth", weights_only=True, map_location=device))

grading_model = GradingModel()
grading_model.to(device)
grading_model.load_state_dict(torch.load("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/models/classification/grading_model_pretrained.pth", weights_only=True, map_location=device))

segmentation_model = UNet(3, 5)
segmentation_model.to(device)
segmentation_model.load_state_dict(torch.load("/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/models/segmentation/pretrained/unet_pretrained.pth", weights_only=True, map_location=device))

optimizer = torch.optim.Adam(grading_model.parameters(), lr=1e-5)

criterion = torch.nn.CrossEntropyLoss()

if TENSORBOARD:
    writer = SummaryWriter(f"runs/{LOG_NAME}")

# Training with mask generator

def validate(grading_model, grading_model_pretrained, segmentation_model, validation_dataloader, criterion, epoch=None):
        validation_loss = 0

        predicted_values = []
        targets = []

        grading_model.eval()
        grading_model_pretrained.eval()
        segmentation_model.eval()
        with torch.no_grad():
            for batch_index, (input_batch, target_batch) in tqdm(enumerate(validation_dataloader)):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                masks = segmentation_model(input_batch)
                pretrained_logits, pretrained_f_low, pretrained_f_high, _ = grading_model_pretrained(input_batch)

                logits, _, _, attention_maps = grading_model(input_batch, masks, pretrained_f_low, pretrained_f_high)

                if epoch is not None and epoch % 1 == 0 and batch_index == 0:
                    reference_image = input_batch[0].cpu().detach().numpy()
                    reference_image = reference_image.transpose(1, 2, 0)

                    # TODO: Add min max scaling
                    reference_image = (reference_image - reference_image.min()) / (reference_image.max() - reference_image.min())
                    reference_image = (reference_image * 255).astype('uint8')
                    if not os.path.exists("output_masks"):
                        os.makedirs("output_masks")
                    cv2.imwrite("output_masks/reference_image.png", reference_image)
                    
                    for i in range(5):
                        attention_map = attention_maps[0][i].cpu().detach().numpy()
                        # mask = mask.transpose(1, 2, 0)
                        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
                        attention_map = (attention_map * 255).astype('uint8')

                        mask = masks[0][i].cpu().detach().numpy()
                        mask = (mask*255).astype('uint8')

                        output_path = f"output_masks/mask_{i}/"
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        cv2.imwrite(os.path.join(output_path, f"mask_{epoch}.png"), attention_map)
                        cv2.imwrite(os.path.join(output_path, f"mask_generator.png"), mask)
                
                loss = criterion(logits, target_batch)
                normalized_output = torch.softmax(logits, dim=-1)

                predicted_values.extend(normalized_output.cpu().detach().tolist())
                
                
                if len(target_batch) > 1:
                    targets += target_batch.squeeze().cpu().detach().tolist()
                else:
                    targets.append(target_batch[0].cpu().detach().item())

                validation_loss += loss.detach().item()

                del input_batch, target_batch, masks, logits, attention_maps, loss, normalized_output
                torch.cuda.empty_cache()

        mean_validation_loss = validation_loss / len(validation_dataloader)

        predicted_values = torch.tensor(predicted_values)
        targets = torch.tensor(targets)

        f1_metric = MulticlassF1Score(num_classes=5, average='macro')
        f1_metric.update(predicted_values, targets)
        f1_score = f1_metric.compute()

        accuracy_metric = MulticlassAccuracy(num_classes=5, average='macro')
        accuracy_metric.update(predicted_values, targets)
        accuracy_score = accuracy_metric.compute()
        
        auprc_metric = MulticlassAUPRC(num_classes=5, average='macro')
        auprc_metric.update(predicted_values, targets)
        auprc_score = auprc_metric.compute()

        auroc_metric = MulticlassAUROC(num_classes=5, average='macro')
        auroc_metric.update(predicted_values, targets)
        auroc_score = auroc_metric.compute()

        return mean_validation_loss, accuracy_score, f1_score, auprc_score, auroc_score

def train(grading_model, grading_model_pretrained, segmentation_model, train_dataloader, validation_dataloader, optimizer, criterion, n_epochs):
    best_validation_loss = float("inf")
    best_model_state_dict = None
    for epoch in range(n_epochs):
        epoch_loss = 0

        grading_model.train()
        segmentation_model.eval()
        grading_model_pretrained.eval()
        for input_batch, target_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                masks = segmentation_model(input_batch)
                pretrained_logits, pretrained_f_low, pretrained_f_high, _ = grading_model_pretrained(input_batch)
                
                masks = masks.detach()
                pretrained_f_low = pretrained_f_low.detach()
                pretrained_f_high = pretrained_f_high.detach()

            logits, _, _, attention_maps = grading_model(input_batch, masks, pretrained_f_low, pretrained_f_high)

            loss = criterion(logits, target_batch)

            epoch_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

        del input_batch
        del target_batch
        # del masks
        del logits
        del attention_maps
        torch.cuda.empty_cache()
        
        _, train_accuracy_score, train_f1_score, train_auprc_score, train_auroc_score = validate(grading_model, grading_model_pretrained, segmentation_model, train_metrics_dataloader, criterion)
        mean_validation_loss, validation_accuracy_score, validation_f1_score, validation_auprc_score, validation_auroc_score = validate(grading_model, grading_model_pretrained, segmentation_model, validation_dataloader, criterion, epoch)

        mean_training_loss = epoch_loss / len(train_dataloader) / BATCH_SIZE

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            torch.save(grading_model.state_dict(), "grading_model_with_masks_ckpt.pth")
            best_model_state_dict = grading_model.state_dict()

        if TENSORBOARD:
            writer.add_scalar("train/Loss", mean_training_loss, epoch)
            writer.add_scalar("train/Accuracy", train_accuracy_score, epoch)
            writer.add_scalar("train/F1 Score", train_f1_score, epoch)
            writer.add_scalar("train/AUPRC", train_auprc_score, epoch)
            writer.add_scalar("train/AUROC", train_auroc_score, epoch)

            writer.add_scalar("validation/Loss", mean_validation_loss, epoch)
            writer.add_scalar("validation/Accuracy", validation_accuracy_score, epoch)
            writer.add_scalar("validation/F1 Score", validation_f1_score, epoch)
            writer.add_scalar("validation/AUPRC", validation_auprc_score, epoch)
            writer.add_scalar("validation/AUROC", validation_auroc_score, epoch)

        if MLFLOW:
            mlflow.log_metric("train/Loss", mean_training_loss, epoch)
            mlflow.log_metric("train/Accuracy", train_accuracy_score, epoch)
            mlflow.log_metric("train/F1 Score", train_f1_score, epoch)
            mlflow.log_metric("train/AUPRC", train_auprc_score, epoch)
            mlflow.log_metric("train/AUROC", train_auroc_score, epoch)
            mlflow.log_metric("validation/Loss", mean_validation_loss, epoch)
            mlflow.log_metric("validation/Accuracy", validation_accuracy_score, epoch)
            mlflow.log_metric("validation/F1 Score", validation_f1_score, epoch)
            mlflow.log_metric("validation/AUPRC", validation_auprc_score, epoch)
            mlflow.log_metric("validation/AUROC", validation_auroc_score, epoch)
            
        print(f"Epoch: {epoch}, Mean training loss: {mean_training_loss}, Mean validation loss: {mean_validation_loss}")

    grading_model.load_state_dict(best_model_state_dict)

train(grading_model, grading_model_pretrained, segmentation_model, train_dataloader, validation_dataloader, optimizer, criterion, 100)

torch.save(grading_model.state_dict(), "grading_model_fine_tuned.pth")

mlflow.pytorch.log_model(grading_model)


