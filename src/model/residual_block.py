import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.heads = nn.ModuleList()
        self.head_weights = nn.Parameter(torch.ones(heads))

        kernels = [3, 5, 7, 9]
        out_channels_list = [8, 16, 32, 64]
        activations = [nn.ReLU, nn.GELU, nn.SiLU, nn.ELU]

        for i in range(heads):
            self.heads.append(
                ConvResidualBlock(
                    in_channels,
                    out_channels=out_channels_list[i % len(out_channels_list)],
                    kernel_size=kernels[i % len(kernels)],
                    padding=kernels[i % len(kernels)] // 2,
                    activation=activations[i % len(activations)]
                )
            )
        self.out_channels = max(out_channels_list)

        if in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm1d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        head_outputs = []

        for head in self.heads:
            out = head(x)
            
            if out.shape[1] != self.out_channels:
                pad = self.out_channels - out.shape[1]
                out = F.pad(out, (0, 0, 0, pad))  
            head_outputs.append(out)

        stacked = torch.stack(head_outputs, dim=0) 
        weights = F.softmax(self.head_weights, dim=0).view(-1, 1, 1, 1)
        out = (stacked * weights).sum(dim=0)
        return self.norm(out + identity)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out += identity
        return self.activation(out)

