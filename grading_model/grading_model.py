import torch
from torch import nn


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
        
    # Separate function for getting from f_low to f_high and logits so that later instead of f_low I can pass f_low * pseudo_mask
    def logits_from_flow(self, f_low):
        pre_f_high = self.post_f_low_seq(f_low).mean((-2, -1)) # Global average pooling to convert 1024x20x20 to 1024x1x1

        logits = self.fc(pre_f_high)
        f_high = self.conv1x1(pre_f_high[..., None, None]) # pre_f_high[..., None, None] is basically double unsqueeze on last dims
        return logits, f_high

    def forward(self, x):
        f_low = self.f_low_seq(x)
        return self.logits_from_flow(f_low)
