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


def read_trajectories(directories, label):
    if not isinstance(directories, list):
        directories = [directories]  # if there is only one directory
    traj_data = []

    x_range = [np.inf, -np.inf]
    y_range = [np.inf, -np.inf]
    z_range = [np.inf, -np.inf]

    files_to_use = []
    for dir in directories:
        if dir.endswith(".json"):
            # Direct path to a trajectory
            files_to_use.append(dir)
        else:
            files = os.listdir(dir)
            if 'sets.json' in files:
                files.remove('sets.json')
            files = [os.path.join(dir, file) for file in files]
            files_to_use += files

    # For each episode
    for filename in files_to_use:
        if filename == "sets.json":
            continue
        traj_data.append({})
        traj_data[-1]["obs"] = []
        traj_data[-1]["label"] = label

        with open(filename) as main_file:
            for line in main_file:
                step = json.loads(line)
                key = list(step.keys())[0]

                # Normalize x,y,z location of player/agent as obs
                player_pos = step[key]["Observations"]["Players"][0]["Position"][0]

                x = player_pos["X"]
                y = player_pos["Y"]
                z = player_pos["Z"]
                obs_list = normalize_pos([x, y, z])

                if x > x_range[1]:
                    x_range[1] = x
                if x < x_range[0]:
                    x_range[0] = x

                if y > y_range[1]:
                    y_range[1] = y
                if y < y_range[0]:
                    y_range[0] = y

                if z > z_range[1]:
                    z_range[1] = z
                if z < z_range[0]:
                    z_range[0] = z

                traj_data[-1]["obs"].append(obs_list)

    ranges = {"x_range": x_range, "y_range": y_range, "z_range": z_range}
    return traj_data, ranges


def normalize_pos(pos):
    normalized_x = (pos[0] + 126.5697) / 2995.3220
    normalized_y = (pos[1] - 10903.4404) / 10060.0283
    normalized_z = (pos[2] + 313.1935) / 880.6552
    return [normalized_x, normalized_y, normalized_z]


class TrajectoryDatasetSymbolic(Dataset):
    def __init__(self, human_dirs, agent_dirs, seq_length):
        # Label Human trajectories 1.0, Agent trajectories 0.0
        self.data = read_trajectories(human_dirs, 1.0)[
            0] + read_trajectories(agent_dirs, 0.0)[0]
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
