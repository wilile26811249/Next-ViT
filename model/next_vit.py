import torch
import torch.nn as nn

from .module import Stem, PatchEmbed, Block


class NextVit(nn.Module):
    def __init__(self, in_channels = 3, stage3_repeat = 2, num_class = 1000, drop_path = 0.):
        super().__init__()
        self.next_vit_channel = [96, 192, 384, 768]

        # Next-Vit Layer
        self.stem = Stem(in_channels, 64)
        self.stage1 = nn.Sequential(
            PatchEmbed(64, self.next_vit_channel[0]),
            Block(self.next_vit_channel[0], self.next_vit_channel[0], 1, 0, 1, 8, drop_path),
        )
        self.stage2 = nn.Sequential(
            PatchEmbed(self.next_vit_channel[0], self.next_vit_channel[1]),
            Block(self.next_vit_channel[1], 256, 3, 1, 1, 4, drop_path),
        )
        self.stage3 = nn.Sequential(
            PatchEmbed(256, self.next_vit_channel[2]),
            Block(self.next_vit_channel[2], 512, 4, 1, stage3_repeat, 2, drop_path),
        )
        self.stage4 = nn.Sequential(
            PatchEmbed(512, self.next_vit_channel[3]),
            Block(self.next_vit_channel[3], 1024, 2, 1, 1, 1, drop_path),
        )

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(1024, 1280),
            nn.ReLU(inplace = True),
        )

        # Final Classifier
        self.classifier = nn.Linear(1280, num_class)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        logit = self.classifier(x)
        return logit


def NextViT_S():
    net = NextVit(stage3_repeat = 2, drop_path = 0.1)
    return net

def NextViT_B():
    net = NextVit(stage3_repeat = 4, drop_path = 0.2)
    return net

def NextViT_L():
    net = NextVit(stage3_repeat = 6, drop_path = 0.2)
    return net