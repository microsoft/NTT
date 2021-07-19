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

import torchvision
import torch
import torch.nn as nn


class TopdownClassifier(nn.Module):
    def __init__(self, device, dropout=0.0, hidden_size=0):
        super(TopdownClassifier, self).__init__()
        self.device = device

        # Define topdown model
        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        # freeze convolution weights
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # replace last layer with a new layer with only 2 outputs
        self.num_features = self.vgg16.classifier[6].in_features
        self.features = list(
            self.vgg16.classifier.children())[
            :-2]  # Remove last layer
        self.features.extend([nn.Dropout(dropout)])
        if hidden_size > 0:
            self.features.extend([nn.Linear(self.num_features, hidden_size)])
            self.num_features = hidden_size
        # Add our layer with 2 outputs
        self.features.extend([nn.Linear(self.num_features, 2)])
        self.vgg16.classifier = nn.Sequential(
            *self.features)  # Replace the model classifier

        self.model = self.vgg16.to(device)

    def forward(self, x):
        return self.model(x)

    def loss_function(self, x, y):
        return nn.CrossEntropyLoss()(x, y.long())

    def correct_predictions(self, model_output, labels):
        _, predictions = torch.max(model_output.data, 1)
        return (predictions == labels).sum().item()

    def load_state_dict(self, f):
        self.model.load_state_dict(f)
