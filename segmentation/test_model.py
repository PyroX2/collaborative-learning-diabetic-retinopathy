import torch
from segmentation.dataset import DRSegmentationDataset
from segmentation.unet import UNet
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset_dir = '/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/datasets/processed_segmentation_dataset/test_set'
model_path = '/users/scratch1/s189737/collaborative-learning-diabetic-retinopathy/models/segmentation/segmentation_generator.pth'

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


test_dataset = DRSegmentationDataset(test_dataset_dir)

test_dataloader = torch.utils.data.DataLoader(
                      test_dataset, 
                      batch_size=1)

loaded_model = UNet(3, NUM_CLASSES)
loaded_model.to(device)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))

loaded_model.eval()
loss = torch.nn.BCELoss()

class_names = test_dataset.class_names

metrics, validation_epoch_loss, val_auroc, val_auprc, val_accuracy, val_f1_score = calculate_mask_metrics(test_dataloader, loaded_model, loss, class_names)

metrics_names = ['loss', 'auroc', 'auprc', 'accuracy', 'f1_score']

plt.figure(figsize=(15, 8))
for i, metric in enumerate(metrics_names):
    plt.subplot(2, 3, i+1)
    values = [m[metric] for m in metrics]
    bars = plt.bar(class_names, values)
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    plt.ylim(0, 1.2 if metric != 'loss' else max(values)*1.2)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

plt.subplot(2, 3, 6)
values = [validation_epoch_loss, val_auroc, val_auprc, val_accuracy, val_f1_score]
bars = plt.bar(['Test Loss', 'Test AUROC', 'Test AUPRC', 'Test Accuracy', 'Test F1 Score'], values)
plt.title('Overall Metrics')
plt.xticks(rotation=45)
plt.ylabel('Value')
plt.ylim(0, 1.2)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{value:.3f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()
plt.savefig('segmentation_metrics.png')