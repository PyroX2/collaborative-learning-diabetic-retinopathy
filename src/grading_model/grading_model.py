import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class GradingModel(nn.Module):
    def __init__(self, num_outputs=5, num_lesions=4):
        super().__init__()

        self.num_lesions = num_lesions

        self.f_low_seq = nn.Sequential(nn.Conv2d(3, 16, (3, 3), padding='same'),
                                       nn.ReLU(),
                                       torch.nn.BatchNorm2d(16),
                                       nn.Conv2d(16, 32, (3, 3), padding="same"),
                                       nn.ReLU(),
                                       torch.nn.BatchNorm2d(32)
                                    )

        self.post_f_low_seq = nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding='same'),
                                            nn.MaxPool2d((2, 2)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(64),
                                            nn.Conv2d(64, 128, (3, 3), padding='same'),
                                            nn.MaxPool2d((2, 2)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(128),
                                            nn.Conv2d(128, 256, (3, 3), padding='same'),
                                            nn.MaxPool2d((2, 2)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(256),
                                            nn.Conv2d(256, 512, (3, 3), padding='same'),
                                            nn.MaxPool2d((2, 2)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(512),
                                            nn.Conv2d(512, 1024, (3, 3), padding='same'),
                                            nn.MaxPool2d((2, 2)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(1024)
                                        )
        
        self.fc = nn.Linear(1024, num_outputs)

        # Uses grouping so that each lesion has its own set of weights
        self.mask_preprocess_seq = nn.Sequential(nn.Conv2d(num_lesions, num_lesions*8, (3, 3), padding='same', groups=num_lesions),
                                                 nn.ReLU(),
                                                 nn.BatchNorm2d(num_lesions*8),
                                                 nn.Conv2d(num_lesions*8, num_lesions*16, (3, 3), padding='same', groups=num_lesions),
                                                 nn.ReLU(),
                                                 nn.BatchNorm2d(num_lesions*16))
        
        self.mask_input_image_conv = nn.Sequential(nn.Conv2d(num_lesions*48, num_lesions*32, (3, 3), padding='same', groups=num_lesions),
                                                   nn.ReLU())
        
        self.output_conv1x1 = nn.Conv2d(in_channels=num_lesions*1024, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.conv1x1 = nn.Conv2d(
                            in_channels=1024,  # number of input channels
                            out_channels=32,  # number of output channels
                            kernel_size=1,  # 1x1 convolution
                            stride=1,
                            padding=0,
                        )

        self.w_high = nn.Conv2d(num_lesions*32, num_lesions, (3, 3), padding='same', groups=num_lesions) # 1x1 convolution


    # Separate function for getting from f_low to f_high and logits so that later instead of f_low I can pass f_low * pseudo_mask
    def logits_from_flow(self, f_low):
        pre_f_high = self.post_f_low_seq(f_low)

        # Global average pooling
        pre_f_high = pre_f_high.mean((-2, -1)) # Global average pooling to convert 1024x20x20 to 1024x1x1 (this simplifies to 1024 so further unsqueeze is needed)

        # Output for pretraining
        logits = self.fc(pre_f_high) # Transforming 1024 vector to 5 output logits

        # f_high vector that is used for attention maps
        f_high = self.conv1x1(pre_f_high[..., None, None]) # pre_f_high[..., None, None] is basically double unsqueeze on last dims
        return logits, f_high


    def forward(self, x: torch.Tensor, attention_maps: torch.Tensor=None) -> torch.Tensor:
        # Masks should be of size BATCH_SIZE x number_of_masks x height x width
        f_low = self.f_low_seq(x)
        
        f_low_expanded = f_low.unsqueeze(1)
        attention_maps_expanded = attention_maps.unsqueeze(2)
        new_f_low = torch.mul(f_low_expanded, attention_maps_expanded)

        # new_f_low: (batch_size, num_lesions, 32, 640, 640)
        batch_size, num_lesions, c, h, w = new_f_low.shape

        # Merge batch and lesion dims for processing
        new_f_low_reshaped = new_f_low.view(batch_size * self.num_lesions, c, h, w)  # (batch_size*num_lesions, 32, 640, 640)

        # Pass through post_f_low_seq and global average pool
        lesion_cls_output_vector = self.post_f_low_seq(new_f_low_reshaped).mean((-2, -1), keepdim=True)  # (batch_size*num_lesions, 1024, 1, 1)

        # Reshape back to (batch_size, num_lesions*1024, 1, 1)
        all_masks_classification_outputs = lesion_cls_output_vector.view(batch_size, num_lesions*1024, 1, 1)

        pre_logits = self.output_conv1x1(all_masks_classification_outputs)
        logits = self.fc(torch.squeeze(pre_logits, (-2, -1)))  # Squeeze the last two dimensions to get logits of shape [BATCH SIZE x 1024]

        return logits, attention_maps
