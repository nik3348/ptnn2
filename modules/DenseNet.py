import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            )

    def _make_dense_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate

        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(num_blocks):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f"denseblock_{i}", block)
            num_features += num_layers * growth_rate
            if i != len(num_blocks) - 1:
                out_channels = int(num_features * reduction)
                transition = TransitionLayer(num_features, out_channels)
                self.features.add_module(f"transition_{i}", transition)
                num_features = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return out.view(-1)
