from collections import OrderedDict

import torch
import torch.nn as nn


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", bias=False):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)

        self.pointwise_conv = PointwiseConv2d(in_channels, out_channels, 1, 1, "same", bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, use_bias=True):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", use_bias=use_bias)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_shortcut_e1 = PointwiseConv2d(in_channels, features, use_bias, stride=2)

        self.encoder2 = UNet._block(features, features * 2, name="enc2", use_bias=use_bias)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_shortcut_e2 = PointwiseConv2d(features, features*2, use_bias, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", use_bias=use_bias)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_shortcut_e3 = PointwiseConv2d(features*2, features*4, use_bias, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", use_bias=use_bias)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pointwise_shortcut_e4 = PointwiseConv2d(features*4, features*8, use_bias, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", use_bias=use_bias)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(features * 8 * 2, features * 8, name="dec4", use_bias=use_bias)
        self.pointwise_shortcut_d4 = PointwiseConv2d(features * 8 * 2, features*8, use_bias, stride=1)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(features * 4 * 2, features * 4, name="dec3", use_bias=use_bias)
        self.pointwise_shortcut_d3 = PointwiseConv2d(features * 4 * 2, features*4, use_bias, stride=1)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(features * 2 * 2, features * 2, name="dec2", use_bias=use_bias)
        self.pointwise_shortcut_d2 = PointwiseConv2d(features*2*2, features*2, use_bias, stride=1)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1", use_bias=use_bias)
        self.pointwise_shortcut_d1 = PointwiseConv2d(features*2, features, use_bias, stride=1)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool1(enc1) + self.pointwise_shortcut_e1(x)
        enc2 = self.encoder2(enc1_pooled)
        enc2_pooled = self.pool2(enc2) + self.pointwise_shortcut_e2(enc1_pooled)
        enc3 = self.encoder3(enc2_pooled)
        enc3_pooled = self.pool3(enc3) + self.pointwise_shortcut_e3(enc2_pooled)
        enc4 = self.encoder4(enc3_pooled)
        enc4_pooled = self.pool4(enc4) + self.pointwise_shortcut_e4(enc3_pooled)

        bottleneck = self.bottleneck(enc4_pooled)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        return torch.sigmoid(out)

    # One block consists of two convolutional layers followed by batch normalization and ReLU activation
    @staticmethod
    def _block(in_channels, features, name, use_bias):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "separable_conv1",
                        SeparableConv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            stride=1,
                            padding="same",
                            bias=use_bias,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "separable_conv2",
                        SeparableConv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            stride=1,
                            padding="same",
                            bias=use_bias,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, init_features=32, use_bias=True)
    x = torch.rand((1, 3, 640, 640))  # Example input
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 256) for a single-channel output