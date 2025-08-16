from grading_model import GradingModel
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torcheval.metrics import BinaryAccuracy, BinaryAUPRC, BinaryAUROC, BinaryF1Score
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = v2.Compose([
    v2.ToTensor()])


test_root = ""
test_dataset = ImageFolder(test_root, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


def validate(grading_model, validation_dataloader, criterion):
        validation_loss = 0

        predicted_values = []
        targets = []

        grading_model.eval()
        for input_batch, target_batch in tqdm(validation_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device).to(torch.float32)

            logits, f_low, f_high, _ = grading_model(input_batch)
            output = F.sigmoid(logits.squeeze(-1))

            loss = criterion(output, target_batch)

            predicted_values += output.cpu().detach().tolist()

            targets += target_batch.cpu().detach().tolist()

            validation_loss += loss.detach().item()

            del loss
            del input_batch
            del target_batch
            del logits
            del f_high
            del output
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

model = GradingModel(num_lesions=4, num_outputs=1).to(device)
model.load_state_dict(torch.load("", map_location=device))
model.eval()

metrics = validate(model, test_dataloader, nn.BCELoss())
print(f"Validation Loss: {metrics[0]}")
print(f"Accuracy: {metrics[1]}")
print(f"F1 Score: {metrics[2]}")
print(f"AUPRC: {metrics[3]}")
print(f"AUROC: {metrics[4]}")