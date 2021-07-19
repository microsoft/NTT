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

# Script to reproduce Figures 9 and 10 in Section 5.2
# Provided data from table 2 in the appendix
# or generated by evaluate_ANTT_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Data from table 2 in the appendix
df = pd.DataFrame({
    'ANTT Model': ['SYM-FF', 'SYM-GRU', 'VIS-FF', 'VIS-GRU', 'TD-CNN', 'BC-CNN',
                   'SYM-FF', 'SYM-GRU', 'VIS-FF', 'VIS-GRU', 'TD-CNN', 'BC-CNN',
                   'SYM-FF', 'SYM-GRU', 'VIS-FF', 'VIS-GRU', 'TD-CNN', 'BC-CNN'],
    'Comparison': ["Identity Accuracy", "Identity Accuracy", "Identity Accuracy",
                   "Identity Accuracy", "Identity Accuracy", "Identity Accuracy",
                   "Human-Agent", "Human-Agent", "Human-Agent",
                   "Human-Agent", "Human-Agent", "Human-Agent",
                   "Hybrid-Symbolic", "Hybrid-Symbolic", "Hybrid-Symbolic",
                   "Hybrid-Symbolic", "Hybrid-Symbolic", "Hybrid-Symbolic"],
    'Accuracy': [0.85, 0.85, 0.633, 0.767, 0.583, 0.717,
                 0.85, 0.85, 0.633, 0.767, 0.583, 0.717,
                 0.475, 0.400, 0.225, 0.425, 0.525, 0.475],
    'std': [0.062, 0.082, 0.041, 0.097, 0.075, 0.145,
            0.062, 0.082, 0.041, 0.097, 0.075, 0.145,
            0.166, 0.200, 0.050, 0.127, 0.094, 0.050]})

# Bootstrap observations to get std bars
dfCopy = df.copy()
duplicates = 3000  # increase this number to increase precision
for _, row in df.iterrows():
    for times in range(duplicates):
        new_row = row.copy()
        new_row['Accuracy'] = np.random.normal(row['Accuracy'], row['std'])
        dfCopy = dfCopy.append(new_row, ignore_index=True)

sns.catplot(
    x="Comparison",
    y="Accuracy",
    hue="ANTT Model",
    kind="bar",
    data=dfCopy)

# Data from table 2 in the appendix
rankdf = pd.DataFrame({
    'ANTT Model': ['SYM-FF', 'SYM-GRU', 'VIS-FF', 'VIS-GRU', 'TD-CNN', 'BC-CNN',
                   'SYM-FF', 'SYM-GRU', 'VIS-FF', 'VIS-GRU', 'TD-CNN', 'BC-CNN'],
    'Comparison': ["Human-Agent", "Human-Agent", "Human-Agent",
                   "Human-Agent", "Human-Agent", "Human-Agent",
                   "Hybrid-Symbolic", "Hybrid-Symbolic", "Hybrid-Symbolic",
                   "Hybrid-Symbolic", "Hybrid-Symbolic", "Hybrid-Symbolic"],
    'Spearman Rank Correlation': [0.364, 0.173, -0.041, 0.220, 0.222, -0.009,
                                  -0.244, -0.249, -0.165, -0.056, -0.093, -0.095],
    'std': [0.043, 0.049, 0.160, 0.267, 0.059, 0.131,
            0.252, 0.210, 0.286, 0.331, 0.149, 0.412]})

# Bootstrap observations to get std bars
rankdfCopy = rankdf.copy()
duplicates = 3000  # increase this number to increase precision
for index, row in rankdf.iterrows():
    for times in range(duplicates):
        new_row = row.copy()
        new_row['Spearman Rank Correlation'] = np.random.normal(
            row['Spearman Rank Correlation'], row['std'])
        rankdfCopy = rankdfCopy.append(new_row, ignore_index=True)

sns.catplot(
    x="Comparison",
    y="Spearman Rank Correlation",
    hue="ANTT Model",
    kind="bar",
    data=rankdfCopy)

plt.show()
