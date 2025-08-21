import torch
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
import mlflow
from typing import List, Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
import torch.nn.functional as F


postprocess_transform = v2.Compose([
    v2.Resize([640, 640])
])

def memory_stats() -> None:
    print("Memory allocated:", torch.cuda.memory_allocated()/1024**2)
    print("Memory cached:", torch.cuda.memory_reserved()/1024**2)

def log_class_metrics(all_metrics: List[Dict], step: int, split: str="split_not_specified", log_mlflow: bool=True, tensorboard_writer: SummaryWriter=None) -> None:
    for metrics in all_metrics:
        class_name = metrics['class_name']
        auroc = metrics['auroc']
        auprc = metrics['auprc']
        accuracy = metrics['accuracy']
        f1_score = metrics['f1_score']
        loss = metrics['loss']

        if log_mlflow:
            mlflow.log_metric(f"{class_name}/{split}/AUROC", auroc, step=step)
            mlflow.log_metric(f"{class_name}/{split}/AUPRC", auprc, step=step)
            mlflow.log_metric(f"{class_name}/{split}/Accuracy", accuracy, step=step)
            mlflow.log_metric(f"{class_name}/{split}/F1Score", f1_score, step=step)
            mlflow.log_metric(f"{class_name}/{split}/DiceLoss", loss, step=step)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(f"{class_name}/{split}/AUROC", auroc, step)
            tensorboard_writer.add_scalar(f"{class_name}/{split}/AUPRC", auprc, step)
            tensorboard_writer.add_scalar(f"{class_name}/{split}/Accuracy", accuracy, step)
            tensorboard_writer.add_scalar(f"{class_name}/{split}/F1Score", f1_score, step)
            tensorboard_writer.add_scalar(f"{class_name}/{split}/DiceLoss", loss, step)

def calculate_mask_metrics(validation_px_dataloader, generator_model, criterion, class_names, device=None) -> Tuple:
    num_classes = len(class_names)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = [[] for _ in range(num_classes)]
    predicted_values = [[] for _ in range(num_classes)]

    validation_loss = [0 for _ in range(num_classes)]

    generator_model.eval()
    with torch.no_grad():
        for batch_index, px_level_batch in tqdm(enumerate(validation_px_dataloader)):
            mask_input_tensor = px_level_batch[0].to(device)
            mask_target_tensor = px_level_batch[1].to(device)

            # Generator masks for px level dataset and image level dataset
            generator_output = generator_model(mask_input_tensor)
            generator_output = F.sigmoid(generator_output)
            generator_output = postprocess_transform(generator_output)

            for i in range(num_classes):
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
    for i in range(num_classes):
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