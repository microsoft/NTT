# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2021 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

import torch
from torch import nn
from torchvision import models


class BarcodesClassifier(nn.Module):
    def __init__(self, device, dropout=0.0, hidden=32):
        super(BarcodesClassifier, self).__init__()

        vgg16 = models.vgg16(pretrained=True)

        # freeze convolution weights
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # replace last 2 layers (dropout and fc) with a dropout layer and 1 or
        # 2 fc layers
        num_features = vgg16.classifier[6].in_features
        # Remove last 2 layers (dropout and fc)
        features = list(vgg16.classifier.children())[:-2]
        features.extend([nn.Dropout(dropout)])
        if hidden is not None:
            features.extend([nn.Linear(num_features, hidden)])
            num_features = hidden
        # Add our layer with 2 outputs
        features.extend([nn.Linear(num_features, 2)])
        vgg16.classifier = nn.Sequential(
            *features)  # Replace the model classifier

        self.model = vgg16
        self.model.to(device)

    def forward(self, x):
        x = self.model(x.to(torch.float))
        return x

    def loss_function(self, x, y):
        return nn.CrossEntropyLoss()(x, y.long())

    def correct_predictions(self, model_output, labels):
        _, predictions = torch.max(model_output.data, 1)
        return (predictions == labels).sum().item()
