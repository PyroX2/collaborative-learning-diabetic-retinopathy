import torch
from torcheval.metrics import BinaryAccuracy, BinaryAUPRC, BinaryAUROC, BinaryF1Score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict


device = "cuda" if torch.cuda.is_available() else "cpu"

def validate(grading_model, segmentation_model, validation_dataloader, criterion, epoch=None):
    validation_loss = 0

    predicted_values = []
    targets = []

    grading_model.eval()
    segmentation_model.eval()
    with torch.no_grad():
        for batch_index, (input_batch, target_batch) in tqdm(enumerate(validation_dataloader)):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device).to(torch.float32)

            masks = segmentation_model(input_batch)
            masks = torch.cat((masks[:, :3], masks[:, 4:]), dim=1) # Drop optic disc if applicable

            logits, attention_maps = grading_model(input_batch, masks)
            output = F.sigmoid(logits).squeeze(-1)
            
            loss = criterion(output, target_batch)

            predicted_values.extend(output.cpu().detach().tolist())
            
            if len(target_batch) > 1:
                targets += target_batch.squeeze().cpu().detach().tolist()
            else:
                targets.append(target_batch[0].cpu().detach().item())

            validation_loss += loss.detach().item()

            del input_batch, target_batch, masks, logits, attention_maps, loss, output
            torch.cuda.empty_cache()

    mean_validation_loss = validation_loss / len(validation_dataloader)

    predicted_values = torch.tensor(predicted_values)
    targets = torch.tensor(targets)

    f1_metric = BinaryF1Score()
    f1_metric.update(predicted_values, targets)
    f1_score = f1_metric.compute()

    accuracy_metric = BinaryAccuracy()
    accuracy_metric.update(predicted_values, targets)
    accuracy_score = accuracy_metric.compute()
    
    auprc_metric = BinaryAUPRC()
    auprc_metric.update(predicted_values, targets)
    auprc_score = auprc_metric.compute()

    auroc_metric = BinaryAUROC()
    auroc_metric.update(predicted_values, targets)
    auroc_score = auroc_metric.compute()

    return mean_validation_loss, accuracy_score, f1_score, auprc_score, auroc_score


def train(models: Dict, dataloaders, optimizers, criterions, update_size=None):
    epoch_loss = 0
    batch_size = 0

    grading_model = models["grading"]
    segmentation_model = models["unet"]

    train_dataloader = dataloaders["img_train"]
    val_dataloader = dataloaders["img_val"]
    train_metrics_dataloader = dataloaders["img_train_metrics"]

    grading_optimizer = optimizers["grading"]
    segmentation_optimizer = optimizers["unet"]

    grading_criterion = criterions["grading"]

    grading_model.train()
    segmentation_model.train()
    for input_index, (input_batch, target_batch) in tqdm(enumerate(train_dataloader)):
        grading_optimizer.zero_grad()
        segmentation_optimizer.zero_grad()

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device).to(torch.float32)

        masks = segmentation_model(input_batch)                
        masks = masks.detach()
        masks = torch.cat((masks[:, :3], masks[:, 4:]), dim=1) # Drop optic disc if applicable

        logits, attention_maps = grading_model(input_batch, masks)
        output = F.sigmoid(logits).squeeze(-1)

        loss = grading_criterion(output, target_batch)

        epoch_loss += loss.detach().item()

        loss.backward()
        grading_optimizer.step()
        segmentation_optimizer.step()

        if batch_size == 0:
            batch_size = input_batch.shape[0]

        del input_batch
        del target_batch
        del masks
        del logits
        del attention_maps
        del output
        torch.cuda.empty_cache()

        if input_index * batch_size >= update_size:
            break
    
    mean_training_loss = epoch_loss / len(train_dataloader) / batch_size
    _, train_accuracy_score, train_f1_score, train_auprc_score, train_auroc_score = validate(grading_model, segmentation_model, train_metrics_dataloader, grading_criterion)
    mean_validation_loss, validation_accuracy_score, validation_f1_score, validation_auprc_score, validation_auroc_score = validate(grading_model, segmentation_model, val_dataloader, grading_criterion)

    metrics = {
        "Loss/train": mean_training_loss,
        "Accuracy/train": train_accuracy_score,
        "F1 Score/train": train_f1_score,
        "AUPRC/train": train_auprc_score,
        "AUROC/train": train_auroc_score,
        "Loss/validation": mean_validation_loss,
        "Accuracy/validation": validation_accuracy_score,
        "F1 Score/validation": validation_f1_score,
        "AUPRC/validation": validation_auprc_score,
        "AUROC/validation": validation_auroc_score
    }
    return metrics