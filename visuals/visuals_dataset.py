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
import json
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.tensor
import base64
from PIL import Image
import io
import itertools


def read_trajectories(directories, label):
    print("Loading trajectories in", directories)
    if not isinstance(directories, list):
        directories = [directories]  # if there is only one directory
    traj_data = []

    files_to_use = []
    for dir in directories:
        files = os.listdir(dir)
        if 'sets.json' in files:
            files.remove('sets.json')
        files = [os.path.join(dir, file) for file in files]
        files_to_use += files

    # For each episode
    for filename in files_to_use:
        traj_data.append({})
        traj_data[-1]["obs"] = []
        traj_data[-1]["label"] = label

        with open(filename) as main_file:
            video = []
            for line in itertools.islice(main_file, 0, None, 10):
                step = json.loads(line)
                key = list(step.keys())[0]

                encoded_img = step[key]["Observations"]["Players"][0]["Image"]["ImageBytes"]
                decoded_image_data = base64.decodebytes(
                    encoded_img.encode('utf-8'))
                image = Image.open(io.BytesIO(decoded_image_data))
                img = np.array(image)
                video.append(img)

            videodata = np.array(video) / 255
            videodata = np.transpose(videodata, (0, 3, 1, 2))
            traj_data[-1]["obs"] = videodata

    print("Files loaded: ", len(traj_data))
    return traj_data


class TrajectoryDatasetVisuals(Dataset):
    def __init__(self, human_dirs, agent_dirs, seq_length):
        # Label Human trajectories 1.0, Agent trajectories 0.0
        self.data = read_trajectories(
            human_dirs, 1.0) + read_trajectories(agent_dirs, 0.0)
        self.seq_length = seq_length

    def __len__(self):
        count = 0
        for episode in self.data:
            count += len(episode["obs"]) // self.seq_length
        return count

    def __getitem__(self, idx):
        count = 0
        for episode in self.data:
            sequences_in_episode = len(episode["obs"]) // self.seq_length
            if idx >= count + sequences_in_episode:
                count += sequences_in_episode
            else:
                sequence_idx_in_episode = idx - count
                sequence_start_idx = sequence_idx_in_episode * self.seq_length
                sample_trajectory = episode["obs"][sequence_start_idx:
                                                   sequence_start_idx + self.seq_length]
                label = episode["label"]
                return torch.tensor(sample_trajectory), torch.tensor(label)
