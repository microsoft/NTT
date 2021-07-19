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

# Script to reproduce Figure 2 in Section 3.3: learning curves of ANTT models

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    path = os.path.abspath('.')  # Run from pwd or specify a path here
    print("Plotting data from ", path)

    # list all subfolders of the folder - each subfolder is considered an experiment and subfolders
    # within that subfolder are separate runs of that experiment
    list_subfolders_with_paths = [
        f.path for f in os.scandir(path) if f.is_dir()]
    print("Found following experiments: ", list_subfolders_with_paths)

    experiment_names = []
    colours = ['red', 'green', 'blue', 'orange', 'pink', 'yellow', 'black']
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=False)
    for experiment, color in zip(list_subfolders_with_paths, colours):
        print('{} = {}'.format(color, experiment))
        run_cvss = [f.path for f in os.scandir(experiment)]
        experiment_name = os.path.basename(os.path.normpath(experiment))
        experiment_names.append(experiment_name)

        run_dfs = []
        for run in run_cvss:
            run_data_frame = pd.read_csv(run)
            run_dfs.append(run_data_frame)

        experiment_df = pd.concat(run_dfs)
        sns.lineplot(
            ax=axes[0][0],
            data=experiment_df,
            x='epoch',
            y='train_loss',
            ci='sd',
            legend='brief',
            label=experiment_name)
        sns.lineplot(
            ax=axes[0][1],
            data=experiment_df,
            x='epoch',
            y='train_acc',
            ci='sd',
            legend='brief',
            label=experiment_name)
        sns.lineplot(
            ax=axes[1][0],
            data=experiment_df,
            x='epoch',
            y='val_loss',
            ci='sd',
            legend='brief',
            label=experiment_name)
        sns.lineplot(
            ax=axes[1][1],
            data=experiment_df,
            x='epoch',
            y='val_acc',
            ci='sd',
            legend='brief',
            label=experiment_name)

    plt.show()


if __name__ == '__main__':
    main()
