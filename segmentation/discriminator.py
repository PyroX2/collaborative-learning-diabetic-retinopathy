from torch.nn import Module
import torch


class Discriminator(Module):
    def __init__(self, in_channels=8):
        super().__init__()

        # Initial in_channels is 8 for 3 layers RGB + 5 binary masks
        self.conv_tuple_1 = self.convolution_tuple(in_channels=in_channels, out_channels=32) # 320x320
        self.conv_tuple_2 = self.convolution_tuple(in_channels=32, out_channels=64) # 160x160
        self.conv_tuple_3 = self.convolution_tuple(in_channels=64, out_channels=128) # 80x80
        self.conv_tuple_4 = self.convolution_tuple(in_channels=128, out_channels=256) # 40x40
        self.conv_tuple_5 = self.convolution_tuple(in_channels=256, out_channels=512) # 20x20
        # self.global_avg_pool_layer = torch.nn.AvgPool2d((2, 2))
        self.fully_connected = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_tuple_1(x)
        x = self.conv_tuple_2(x)
        x = self.conv_tuple_3(x)
        x = self.conv_tuple_4(x)
        x = self.conv_tuple_5(x)
        x = x.mean((-2, -1))
        # flattened = torch.flatten(con5, start_dim=1)
        logits = self.fully_connected(x)
        output = self.sigmoid(logits)

        return output

    @staticmethod
    def convolution_tuple(in_channels, out_channels):
        convolution_tuple_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same'),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels)
        )

        return convolution_tuple_layer