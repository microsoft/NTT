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
from torch.nn import functional as F


class SymbolicClassifier(nn.Module):
    def __init__(self, device, dropout=0, hidden_size=32):
        super(SymbolicClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.gru_enc = nn.GRU(input_size=3,  # xyz location
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              batch_first=True)
        for name, param in self.gru_enc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        hidden_state = torch.zeros(
            (1, batch_size, self.hidden_size), requires_grad=True).to(
            self.device)
        _, final_gru_output = self.gru_enc(x, hidden_state)
        final_gru_output = nn.Dropout(p=self.dropout)(final_gru_output)
        return torch.sigmoid(self.fc2(final_gru_output))

    def loss_function(self, x, y):
        # x has shape (1, batch_size, 1)
        # y has shape (batch_size)
        return F.binary_cross_entropy(x[0, :, 0], y)

    def correct_predictions(self, model_output, labels):
        predictions = (model_output[0, :, 0] > 0.5).float()
        return (predictions == labels).sum().item()
