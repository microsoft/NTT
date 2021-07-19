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
import torch.utils.data
from torch import nn
from torchvision import models


class VisualsClassifier(nn.Module):
    def __init__(self, device, dropout=0.5, hidden_dim=32):
        super(VisualsClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        vgg16 = models.vgg16(pretrained=True)
        # freeze convolution weights
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # replace last layer with a new layer with only 2 outputs
        self.num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1]  # Remove last layer
        vgg16.classifier = nn.Sequential(
            *features)  # Replace the model classifier

        self.cnn_encoder = vgg16

        self.gru_enc = nn.GRU(input_size=self.num_features,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              batch_first=True)

        for name, param in self.gru_enc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        hidden_state = torch.zeros(
            (1, batch_size, self.hidden_dim), requires_grad=True).to(
            self.device)
        # just treat sequence data as batch for the vgg model
        original_shape = x.size()
        x = torch.reshape(
            x,
            (original_shape[0] * original_shape[1],
             original_shape[2],
             original_shape[3],
             original_shape[4])).to(
            torch.float)
        x = self.cnn_encoder(x)
        x = self.dropout(x)
        # convert back to batch,sequence,obs
        x = torch.reshape(
            x,
            (original_shape[0],
             original_shape[1],
             self.num_features))
        _, final_gru_output = self.gru_enc(x, hidden_state)
        final_gru_output = torch.squeeze(final_gru_output, 0)
        final_gru_output = self.dropout(final_gru_output)
        return self.fc2(final_gru_output)

    def loss_function(self, x, y):
        return nn.CrossEntropyLoss()(x, y.long())

    def correct_predictions(self, model_output, labels):
        _, predictions = torch.max(model_output.data, 1)
        return (predictions == labels).sum().item()
