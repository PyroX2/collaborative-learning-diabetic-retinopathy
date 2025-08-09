import torch
from torch import nn


torch.manual_seed(0)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, smooth=1):
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        
        # Compute Dice Coefficient
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Return Dice Loss
        return 1 - dice.mean()
    

if __name__ == "__main__":
    dice_loss = DiceLoss()

    pred = torch.randint(0, 2, (1, 2, 3, 3))
    target = torch.randint(0, 2, (1, 2, 3, 3))

    print("Pred", pred)
    print('Target', target)

    loss = dice_loss(pred, target)
    print(loss)
