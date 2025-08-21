from grading_model.dataset import GradingDataset
from grading_model.grading_model import GradingModel
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import mlflow
import torch.nn.functional as F


# Set manual seed for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


BATCH_SIZE = 64
LEARNING_RATE = 1e-5

USE_MLFLOW = False
USE_TENSORBOARD = False
LOG_NAME = "test"

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = v2.Compose([
    v2.Resize((640, 640)),
    v2.ToTensor()])

if USE_MLFLOW:
    mlflow.set_tracking_uri("http://localhost:5000")
if USE_TENSORBOARD:
    writer = SummaryWriter(f"runs/{LOG_NAME}")

train_root = "train/two_classes"
validation_root = "val/two_classes"
test_root = "test/two_classes"

train_dataset = ImageFolder(train_root, transform=transform)
validation_dataset = ImageFolder(validation_root, transform=transform)

_, train_metrics_dataset = random_split(train_dataset, [0.95, 0.05])

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=42)
validation_dataloader = DataLoader(validation_dataset, BATCH_SIZE, shuffle=True, num_workers=42)
train_metrics_dataloader = DataLoader(train_metrics_dataset, BATCH_SIZE, shuffle=False, num_workers=42)

grading_model = resnet50(weights=ResNet50_Weights.DEFAULT)
grading_model.fc = nn.Linear(grading_model.fc.in_features, 1)
grading_model.to(device)

optimizer = torch.optim.Adam(grading_model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.CrossEntropyLoss()

def validate(grading_model, validation_dataloader, criterion):
        validation_loss = 0

        predicted_values = []
        targets = []

        grading_model.eval()
        for input_batch, target_batch in tqdm(validation_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = grading_model(input_batch)

            loss = criterion(logits, target_batch)

            predicted_values += logits.cpu().detach().tolist()

            targets += target_batch.cpu().detach().tolist()

            validation_loss += loss.detach().item()

            del loss, input_batch, target_batch, logits
            torch.cuda.empty_cache()

        mean_validation_loss = validation_loss / len(validation_dataloader)

        predicted_values = torch.tensor(predicted_values)
        targets = torch.tensor(targets)

        f1_metric = MulticlassF1Score(num_classes=5)
        f1_metric.update(predicted_values, targets)
        f1_score = f1_metric.compute()

        accuracy_metric = MulticlassAccuracy(num_classes=5)
        accuracy_metric.update(predicted_values, targets)
        accuracy_score = accuracy_metric.compute()
        
        auprc_metric = MulticlassAUPRC(num_classes=5)
        auprc_metric.update(predicted_values, targets)
        auprc_score = auprc_metric.compute()

        auroc_metric = MulticlassAUROC(num_classes=5)
        auroc_metric.update(predicted_values, targets)
        auroc_score = auroc_metric.compute()

        return mean_validation_loss, accuracy_score, f1_score, auprc_score, auroc_score

def train(grading_model, train_dataloader, validation_dataloader, optimizer, criterion, n_epochs):
    if USE_MLFLOW:
        mlflow.log_params({"Batch size": BATCH_SIZE, "Learning rate": LEARNING_RATE})

    best_validation_loss = float("inf")
    for epoch in range(n_epochs):
        training_loss = 0
        grading_model.train()
        for input_batch, target_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = grading_model(input_batch)

            loss = criterion(logits, target_batch)
            training_loss += loss.detach().item()
            loss.backward()
            optimizer.step()

        del input_batch, target_batch, logits, loss
        torch.cuda.empty_cache()

        mean_training_loss = training_loss / len(train_dataloader) / BATCH_SIZE

        _, train_accuracy_score, train_f1_score, train_auprc_score, train_auroc_score = validate(grading_model, train_metrics_dataloader, criterion)
        mean_validation_loss, validation_accuracy_score, validation_f1_score, validation_auprc_score, validation_auroc_score = validate(grading_model, validation_dataloader, criterion)

        if USE_TENSORBOARD:
            writer.add_scalar("Loss/train", mean_training_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy_score, epoch)
            writer.add_scalar("F1 Score/train", train_f1_score, epoch)
            writer.add_scalar("AUPRC/train", train_auprc_score, epoch)
            writer.add_scalar("AUROC/train", train_auroc_score, epoch)
            writer.add_scalar("Loss/validation", mean_validation_loss, epoch)
            writer.add_scalar("Accuracy/validation", validation_accuracy_score, epoch)
            writer.add_scalar("F1 Score/validation", validation_f1_score, epoch)
            writer.add_scalar("AUPRC/validation", validation_auprc_score, epoch)
            writer.add_scalar("AUROC/validation", validation_auroc_score, epoch)

        if USE_MLFLOW:
            mlflow.log_metric("Loss/train", mean_training_loss, epoch)
            mlflow.log_metric("Accuracy/train", train_accuracy_score, epoch)
            mlflow.log_metric("F1 Score/train", train_f1_score, epoch)
            mlflow.log_metric("AUPRC/train", train_auprc_score, epoch)
            mlflow.log_metric("AUROC/train", train_auroc_score, epoch)
            mlflow.log_metric("Loss/validation", mean_validation_loss, epoch)
            mlflow.log_metric("Accuracy/validation", validation_accuracy_score, epoch)
            mlflow.log_metric("F1 Score/validation", validation_f1_score, epoch)
            mlflow.log_metric("AUPRC/validation", validation_auprc_score, epoch)
            mlflow.log_metric("AUROC/validation", validation_auroc_score, epoch)

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            torch.save(grading_model.state_dict(), f"models/checkpoints/classification/{LOG_NAME}_best.pth")
            torch.save(optimizer.state_dict(), f"models/checkpoints/classification/{LOG_NAME}_optimizer_best.pth")

        print(f"Epoch: {epoch}, Mean training loss: {mean_training_loss}, Mean validation loss: {mean_validation_loss}")

if USE_MLFLOW:
    with mlflow.start_run(run_name=LOG_NAME):
        train(grading_model, train_dataloader, validation_dataloader, optimizer, criterion, 100)
else:
    train(grading_model, train_dataloader, validation_dataloader, optimizer, criterion, 100)

torch.save(grading_model.state_dict(), f"models/classification/{LOG_NAME}_last.pth")
torch.save(optimizer.state_dict(), f"models/classification/{LOG_NAME}_optimizer_last.pth")


