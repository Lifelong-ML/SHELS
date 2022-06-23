import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, output_dim = 9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,  padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,  padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3,  padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3,  padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 256),     ## 17
            nn.ReLU(inplace = True),
            #nn.Linear(512,128),          ## 19
            #nn.ReLU(inplace = True),
            nn.Linear(256, output_dim),  ## 21
            # nn.ReLU(inplace = True),
            # nn.Linear(32, output_dim) ## 23
        )

    def forward(self, x):
        layer_output = []
        # t = x
        feats = x
        for layer in self.features:
            feats = layer(feats)
            layer_output.append(feats)
        # feats = self.features(x)
        out = feats
        for layer in self.classifier:
            out = layer(out)
            layer_output.append(out)

        return layer_output, out
