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

# Training script for classification (human vs scripted) of Bleeding Edge
# gameplay data

from __future__ import print_function
import argparse
import csv
import os
import time
import torch
import torch.utils.data
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_model, get_dataset


def train(
        model,
        optimizer,
        epoch,
        train_loader,
        log_interval,
        device,
        writer,
        update_model):
    if update_model:
        model.train()
    else:
        model.eval()
    train_loss = 0
    train_running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        obs = data[0].to(device)
        model_output = model(obs)

        label = data[1].to(device)
        loss = model.loss_function(model_output, label)
        train_loss += loss.item()

        train_running_correct += model.correct_predictions(model_output, label)

        if update_model:
            loss.backward()
            optimizer.step()

        if batch_idx + 1 % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data[0])))

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_loader.dataset)
    print('====> Epoch: {} Average Train loss: {:.4f} - Accuracy: {:.2f}'.format(epoch,
          avg_train_loss, train_accuracy))
    if writer is not None:
        writer.add_scalar('Train/LOSS', avg_train_loss, epoch)
        writer.add_scalar('Train/ACCURACY', train_accuracy, epoch)
    return avg_train_loss, train_accuracy


def test(model, optimizer, epoch, test_loader, device, writer):
    model.eval()
    test_loss = 0
    test_running_correct = 0
    with torch.no_grad():
        for data in test_loader:
            optimizer.zero_grad()

            obs = data[0].to(device)
            model_output = model(obs)

            label = data[1].to(device)
            loss = model.loss_function(model_output, label)
            test_loss += loss.item()

            test_running_correct += model.correct_predictions(
                model_output, label)

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100. * test_running_correct / len(test_loader.dataset)
    print('====> Epoch: {} Test set loss: {:.4f} - Accuracy: {:.2f}'.format(epoch,
          test_loss, test_accuracy))
    if writer is not None:
        writer.add_scalar('Test/LOSS', test_loss, epoch)
        writer.add_scalar('Test/ACCURACY', test_accuracy, epoch)
    return test_loss, test_accuracy


def training_loop(
        log_dir,
        run_name,
        model,
        optimizer,
        train_loader,
        test_loader,
        log_interval,
        device,
        total_epochs):
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    # Start Tensorboard Logging
    os.makedirs(os.path.join(log_dir, run_name, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, run_name, "models"), exist_ok=True)
    writer = SummaryWriter(
        os.path.join(
            log_dir,
            run_name,
            "tensorboard",
            timestamp_str))
    # A CSV file for storing results to plot
    os.makedirs(os.path.join(log_dir, run_name, "csv"), exist_ok=True)
    csv_filename = os.path.join(
        log_dir, run_name, "csv", f"{timestamp_str}.csv")
    csvfile = open(csv_filename, 'w')
    fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    # Main training loop
    best_acc = 0
    test_acc = 0
    for epoch in range(0, total_epochs + 1):
        train_loss, train_acc = train(
            model, optimizer, epoch, train_loader, log_interval, device, writer, epoch != 0)
        test_loss, test_acc = test(
            model, optimizer, epoch, test_loader, device, writer)
        if test_acc > best_acc:
            print("New best model with validation accuracy", test_acc)
            best_acc = test_acc
            torch.save(
                model.state_dict(),
                os.path.join(
                    log_dir,
                    run_name,
                    "models",
                    "best.pt"))
        torch.save(
            model.state_dict(),
            os.path.join(
                log_dir,
                run_name,
                "models",
                "last.pt"))
        csvwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

    csvfile.close()  # Close results csv file
    writer.close()  # Close Tensorboard logging
    return best_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict if trajectory is human or scripted.',
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
        default=10,
        help='Number of batches to wait for before logging training status.')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Path where logs will be saved.')
    parser.add_argument('--sequence-length', type=int, default=5,
                        help='Number of observations input in sequence.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout likelihood in the classifier model.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimensions in the classifier model.')
    parser.add_argument('--model-type', type=str, default='symbolic',
                        choices=["visuals", "symbolic", "topdown", "barcode"])
    parser.add_argument(
        '--human-train',
        type=str,
        help='Path to human train data.')
    parser.add_argument(
        '--human-test',
        type=str,
        help='Path to human test data.')
    parser.add_argument(
        '--agent-train',
        type=str,
        help='Path to agent train data.')
    parser.add_argument(
        '--agent-test',
        type=str,
        help='Path to agent test data.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.human_train is not None, "Human train dataset must be specified with --human-train."
    assert args.human_test is not None,  "Human test dataset must be specified with --human-test."
    assert args.agent_train is not None, "Agent train dataset must be specified with --agent-train."
    assert args.agent_test is not None,  "Agent test dataset must be specified with --agent-test."

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    dataset = get_dataset(args.model_type)
    print("Loading training dataset...")
    train_dataset = dataset(human_dirs=args.human_train,
                            agent_dirs=args.agent_train,
                            seq_length=args.sequence_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    print("Loading testing dataset...")
    test_dataset = dataset(human_dirs=args.human_test,
                           agent_dirs=args.agent_test,
                           seq_length=args.sequence_length)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    print("Initializing model...")
    model = get_model(
        args.model_type)(
        device,
        args.dropout,
        args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{args.model_type}-{time.strftime('%Y%m%d-%H%M%S')}"
    training_loop(
        args.log_dir,
        run_name,
        model,
        optimizer,
        train_loader,
        test_loader,
        args.log_interval,
        device,
        args.epochs)
