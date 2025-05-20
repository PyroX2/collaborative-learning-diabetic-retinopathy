from grading_model import GradingModel
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassF1Score
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.ToTensor()])


test_root = "/home/wilk/diabetic_retinopathy/large_dataset/augmented_resized_V2/test"
test_dataset = ImageFolder(test_root, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


def validate(grading_model, validation_dataloader, criterion):
        validation_loss = 0

        predicted_values = []
        targets = []

        grading_model.eval()
        for input_batch, target_batch in tqdm(validation_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits, f_high = grading_model(input_batch)

            loss = criterion(logits, target_batch)

            normalized_output = torch.softmax(logits, dim=-1)

            if normalized_output.shape[0] == 1:
                predicted_values += normalized_output.cpu().detach().tolist()
            else:
                predicted_values += normalized_output.squeeze().cpu().detach().tolist()

            if len(target_batch.shape) == 1:
                targets += target_batch.cpu().detach().tolist()
            else:
                targets += target_batch.squeeze().cpu().detach().tolist()

            validation_loss += loss.detach().item()

            del loss
            del input_batch
            del target_batch
            del logits
            del f_high
            del normalized_output
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

model = GradingModel().to(device)
model.load_state_dict(torch.load("grading_model_dropout_ckpt.pt"))
model.eval()

metrics = validate(model, test_dataloader, nn.CrossEntropyLoss())
print(f"Validation Loss: {metrics[0]}")
print(f"Accuracy: {metrics[1]}")
print(f"F1 Score: {metrics[2]}")
print(f"AUPRC: {metrics[3]}")
print(f"AUROC: {metrics[4]}")