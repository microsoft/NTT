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

# Training script for 5-fold cross-validation (human vs scripted) of
# Bleeding Edge gameplay data

from __future__ import print_function
import json
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim

from utils import get_model, get_dataset
from train import training_loop

# Record already loaded dataset to not reload again
LOADED_DATASETS = dict()


def validate(
        model_type,
        human_root_dirs,
        agent_root_dirs,
        train_subsets,
        val_subsets,
        log_dir,
        args,
        hp):
    device = torch.device("cuda" if args.cuda else "cpu")
    model_class = get_model(model_type)
    dataset_class = get_dataset(model_type)

    if hp["sequence_length"] in LOADED_DATASETS:
        dataset = LOADED_DATASETS[hp["sequence_length"]]
    else:
        print("Loading dataset...")
        dataset = dataset_class(human_dirs=human_root_dirs,
                                agent_dirs=agent_root_dirs,
                                seq_length=hp["sequence_length"])
        LOADED_DATASETS[hp["sequence_length"]] = dataset

    validation_split = len(val_subsets) / \
        (len(train_subsets) + len(val_subsets))
    one_indices = []
    zero_indices = []
    # The built-in __next__ method does not work for our custom datasets
    # => enumerate() and other form of iterations do not work either (e.g. for d in dataset)
    for i in range(len(dataset)):
        if dataset[i][1] == 0:
            zero_indices.append(i)
        else:
            one_indices.append(i)
    fold_size_ones = int(validation_split * len(one_indices))
    fold_size_zeros = int(validation_split * len(zero_indices))
    assert len(val_subsets) == 1
    val_subset_num = val_subsets[0]
    val_indices = zero_indices[val_subset_num * fold_size_zeros: (val_subset_num + 1) * fold_size_zeros] + \
        one_indices[val_subset_num * fold_size_ones: (val_subset_num + 1) * fold_size_ones]
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)

    model = model_class(device, hp["dropout"], hp["hidden_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    run_name = f"validation-fold-{val_subsets[0]}"
    best_acc, test_acc = training_loop(
        log_dir, run_name, model, optimizer, train_loader, val_loader, args.log_interval, device, args.epochs)

    print(f"{run_name}: Final test accuracy {test_acc}, best accuracy {best_acc}")
    return best_acc


def cross_validation(model_type, human_root_dirs, agent_root_dirs, args):
    log_dir = os.path.join(
        args.log_dir,
        f"{model_type}-crossval-{time.strftime('%Y%m%d-%H%M%S')}")
    total_subsets = [0, 1, 2, 3, 4]
    triedHP = set()
    with open(args.hp_info) as hp_file:
        hp_info = json.load(hp_file)
        allHP = hp_info["allHP"]
        defaultHP = hp_info["defaultHP"]
        hp_order = hp_info["hp_order"]

    totalHPcombinations = 0
    for v in allHP.values():
        if len(v) > 1:
            totalHPcombinations += len(v)
    print("Total hyperparameter combinations:", totalHPcombinations)
    best_acc = 0
    best_hp = defaultHP.copy()

    for hp_key in hp_order:
        hp = best_hp.copy()
        for hp_value in allHP[hp_key]:
            hp[hp_key] = hp_value
            if tuple(hp.values()) in triedHP:
                # This combination of hyperparameters has already been tried
                continue
            print(f"Starting new cross validation with hyperparameters: {hp}")
            accs = []
            sweep_log_dir = os.path.join(
                log_dir, '-'.join(map(str, list(hp.values()))))
            for s in total_subsets:
                print(f"Current cross validation {s+1}/{len(total_subsets)}")
                train_subsets = [i for i in total_subsets if i != s]
                acc = validate(
                    model_type,
                    human_root_dirs,
                    agent_root_dirs,
                    train_subsets,
                    [s],
                    sweep_log_dir,
                    args,
                    hp)
                accs.append(acc)
            mean_acc = sum(accs) / len(accs)
            print(
                "Mean best accuracy across all",
                len(total_subsets),
                "sweeps is",
                mean_acc)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_hp = hp.copy()
                print(
                    "New best mean accuracy of",
                    best_acc,
                    "with hyperparameters",
                    best_hp)
            triedHP.add(tuple(hp.values()))
    print(
        "Final best accuracy",
        best_acc,
        "with best hyperparameters",
        best_hp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='5-fold cross-validation for human-agent discriminator on Bleeding Edge trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10000,
        help='Number of batches to wait for before logging training status.')
    parser.add_argument('--log-dir', type=str, default="logs",
                        help='Path to save logs and models.')
    parser.add_argument('--model-type', type=str, default='symbolic',
                        choices=["visuals", "symbolic", "topdown", "barcode"],
                        help='Name of the classifier to train.')
    parser.add_argument(
        '--human-dirs',
        type=str,
        default=['data/ICML2021-train-data/human'],
        nargs='+',
        help='List of directories to human data.')
    parser.add_argument(
        '--agent-dirs',
        type=str,
        nargs='+',
        help='List of directories to agent data.',
        default=[
            'data/ICML2021-train-data/hybrid/checkpoint_12700',
            'data/ICML2021-train-data/hybrid/checkpoint_11400',
            'data/ICML2021-train-data/symbolic/checkpoint10900',
            'data/ICML2021-train-data/symbolic/checkpoint11600'])
    parser.add_argument(
        '--hp-info',
        type=str,
        default="./hyperparameters.json",
        help="File path with all the hyperparameters info.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cross_validation(args.model_type, args.human_dirs, args.agent_dirs, args)
