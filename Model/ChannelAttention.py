import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.globalAveragePooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fullyConnectedLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=5),
            nn.ELU()
        )
        pass

    def forward(self, input):
        globalAveragePooling = self.globalAveragePooling(input)
        fullyConnectedLayer = self.fullyConnectedLayer(globalAveragePooling).unsqueeze(2).unsqueeze(3)

        return fullyConnectedLayer
        pass
    pass
