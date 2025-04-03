from torch.nn import Module
import torch


class Discriminator(Module):
    def __init__(self, in_channels=8):
        super(Discriminator).__init__()

        self.conv_tuple_1 = self.convolution_tuple(in_channels=in_channels, out_channels=32)
        self.conv_tuple_2 = self.convolution_tuple(in_channels=in_channels, out_channels=64)
        self.conv_tuple_3 = self.convolution_tuple(in_channels=in_channels, out_channels=128)
        self.conv_tuple_4 = self.convolution_tuple(in_channels=in_channels, out_channels=256)
        self.conv_tuple_5 = self.convolution_tuple(in_channels=in_channels, out_channels=512)
        self.max_pool_layer = torch.nn.MaxPool2d()
        self.fully_connected = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        con1 = self.conv_tuple_1(x)
        con2 = self.conv_tuple_2(con1)
        con3 = self.conv_tuple_3(con2)
        con4 = self.conv_tuple_4(con3)
        con5 = self.conv_tuple_5(con4)
        max_pooled = self.max_pool_layer(con5)
        output = self.sigmoid(self.fully_connected(max_pooled))

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