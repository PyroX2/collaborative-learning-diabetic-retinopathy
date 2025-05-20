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
        
        self.fc = nn.Sequential(nn.Linear(1024, 5), nn.Dropout(0.5))

        self.mask_preprocess_seq = nn.Sequential(nn.Conv2d(1, 8, (3, 3), padding='same'),
                                                 nn.ReLU(),
                                                 nn.BatchNorm2d(8),
                                                 nn.Conv2d(8, 16, (3, 3), padding='same'),
                                                 nn.ReLU(),
                                                 nn.BatchNorm2d(16))
        
        self.mask_input_image_conv = nn.Sequential(nn.Conv2d(48, 32, (3, 3), padding='same'),
                                                   nn.ReLU())
        
        self.output_conv1x1 = nn.Conv2d(in_channels=5*1024, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.conv1x1 = nn.Conv2d(
                            in_channels=1024,  # number of input channels
                            out_channels=32,  # number of output channels
                            kernel_size=1,  # 1x1 convolution
                            stride=1,
                            padding=0,
                        )
        # self.w_low = nn.Conv2d(33, 32, (3, 3), padding='same')
        self.w_high = nn.Conv2d(32, 1, (3, 3), padding='same') # 1x1 convolution


    # Separate function for getting from f_low to f_high and logits so that later instead of f_low I can pass f_low * pseudo_mask
    def logits_from_flow(self, f_low):
        pre_f_high = self.post_f_low_seq(f_low).mean((-2, -1)) # Global average pooling to convert 1024x20x20 to 1024x1x1

        # Output for pretraining
        logits = self.fc(pre_f_high) # Transforming 1024x1x1 to 5 output logits

        # f_high vector that is used for attention maps
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
                mask = masks[:, i].unsqueeze(1) # Shape [BATCH SIZE x 1 x 640 x 640]
                mask = self.mask_preprocess_seq(mask) # Shape [BATCH SIZE x 16 x 640 x 640]
                concat_mask = torch.concat((f_low, mask), dim=1)
                concat_mask = self.mask_input_image_conv(concat_mask)
                concat_masks.append(concat_mask)

            concat_masks = torch.stack(concat_masks, dim=1) # Shape [BATCH SIZE x NUMBER OF MASKS x 32 x 640 x 640]

            attention_maps = torch.zeros(concat_masks.shape[0], 5, 640, 640)

            all_masks_classification_outputs = torch.zeros((concat_masks.shape[0], 5*1024, 1, 1), device="cuda" if torch.cuda.is_available() else "cpu")
            for i in range(5):
                f_low_att = concat_masks[:, i]

                attention_map = F.sigmoid(self.w_high(torch.mul(f_low_att, f_high))).squeeze()
                attention_maps[:, i, :, :] = attention_map


                # Element wise multiplication of attention map and f_low
                if len(attention_map.shape) == 3:
                    new_f_low = torch.mul(f_low, attention_map.unsqueeze(1))
                elif len(attention_map.shape) == 2:
                    new_f_low = torch.mul(f_low, attention_map[None, None, ...])


                lesion_cls_output_vector = self.post_f_low_seq(new_f_low).mean((-2, -1))[..., None, None]

                all_masks_classification_outputs[:, 1024*i:1024*(i+1), :, :] = lesion_cls_output_vector

            pre_logits = self.output_conv1x1(all_masks_classification_outputs)
            logits = self.fc(pre_logits.squeeze())

            return logits, attention_maps
