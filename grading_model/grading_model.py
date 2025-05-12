import torch
from torch import nn
import torch.nn.functional as F


class GradingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_low_seq = nn.Sequential(nn.Conv2d(3, 16, (3, 3), padding='same'),
                                       nn.ReLU(),
                                       torch.nn.BatchNorm2d(16),
                                       nn.Conv2d(16, 32, (3, 3), padding="same"),
                                       nn.ReLU(),
                                       torch.nn.BatchNorm2d(32)
                                    )

        self.post_f_low_seq = nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding='same'),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(64, 128, (3, 3), padding='same'),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(128),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(128, 256, (3, 3), padding='same'),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(256),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(256, 512, (3, 3), padding='same'),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(512),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(512, 1024, (3, 3), padding='same'),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(1024),
                                        nn.MaxPool2d((2, 2))
                                    )
        
        self.fc = nn.Linear(1024, 5)

        self.conv1x1 = nn.Conv2d(
                            in_channels=1024,  # number of input channels
                            out_channels=32,  # number of output channels
                            kernel_size=1,  # 1x1 convolution
                            stride=1,
                            padding=0,
                        )
        self.w_low = nn.Conv2d(33, 32, (3, 3), padding='same')
        self.w_high = nn.Conv2d(32, 1, (3, 3), padding='same')

    # Separate function for getting from f_low to f_high and logits so that later instead of f_low I can pass f_low * pseudo_mask
    def logits_from_flow(self, f_low):
        pre_f_high = self.post_f_low_seq(f_low).mean((-2, -1)) # Global average pooling to convert 1024x20x20 to 1024x1x1

        logits = self.fc(pre_f_high)
        f_high = self.conv1x1(pre_f_high[..., None, None]) # pre_f_high[..., None, None] is basically double unsqueeze on last dims
        return logits, f_high

    def forward(self, x, masks=None):
        # Masks should be of size BATCH_SIZE x number_of_masks x height x width
        f_low = self.f_low_seq(x)
        logits, f_high = self.logits_from_flow(f_low)
        if masks is None:
            return logits, None
        else:
            concat_masks = []
            for i in range(5):
                mask = masks[:, i].unsqueeze(1)
                concat_mask = torch.concat((f_low, mask), dim=1)
                concat_masks.append(concat_mask)

            concat_masks = torch.stack(concat_masks, dim=1) # Shape [BATCH SIZE x NUMBER OF MASKS x 32+1 x 640 x 640]

            attention_maps = torch.zeros(concat_masks.shape[0], 5, 640, 640)
            for i in range(5):
                f_low_att = F.relu(self.w_low(concat_masks[:, i]))

                attention_map = F.sigmoid(self.w_high(torch.mul(f_low_att, f_high))).squeeze()
                attention_maps[:, i, :, :] = attention_map

            return logits, attention_maps
