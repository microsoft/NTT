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

import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.tensor
from PIL import Image


def read_trajectories(directories, label):
    files_to_use = []
    if not isinstance(directories, list):
        directories = [directories]  # if there is only one directory
    for dir in directories:
        files = os.listdir(dir)
        if 'sets.json' in files:
            files.remove('sets.json')
        files = [os.path.join(dir, file) for file in files]
        files_to_use += files
    images = []
    # For each episode
    for filename in files_to_use:
        image = Image.open(filename)
        image_data = np.array(image) / 255
        image_data = np.resize(
            image_data, (*image_data.shape, 3))  # add 3 channels
        image_data = np.transpose(image_data, (2, 0, 1))
        images.append({'data': image_data, 'label': label})
    return images


class TrajectoryDatasetTopdown(Dataset):
    def __init__(self, human_dirs, agent_dirs, seq_length=None):
        # seq_length is only used here for compatibility with the other
        # datasets
        assert seq_length is None or seq_length == 1, "Topdown data do not have a sequence length, you may need to remove it from the hyperparameter file."
        # Label Human trajectories 1.0, Agent trajectories 0.0
        self.data = read_trajectories(
            human_dirs, 1.0) + read_trajectories(agent_dirs, 0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(
            self.data[idx]['data'], dtype=torch.float32), torch.tensor(
            self.data[idx]['label'], dtype=torch.float32)
