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

from symbolic.symbolic_classifier import SymbolicClassifier
from visuals.visuals_classifier import VisualsClassifier
from barcodes.barcodes_classifier import BarcodesClassifier
from topdown.topdown_classifier import TopdownClassifier
from symbolic.symbolic_dataset import TrajectoryDatasetSymbolic
from visuals.visuals_dataset import TrajectoryDatasetVisuals
from barcodes.barcode_dataset import TrajectoryDatasetBarcodes
from topdown.topdown_dataset import TrajectoryDatasetTopdown


def get_model(model_type):
    if model_type == "visuals":
        return VisualsClassifier
    if model_type == "symbolic":
        return SymbolicClassifier
    if model_type == "barcode":
        return BarcodesClassifier
    if model_type == "topdown":
        return TopdownClassifier
    raise NotImplementedError


def get_dataset(model_type):
    if model_type == "visuals":
        return TrajectoryDatasetVisuals
    if model_type == "symbolic":
        return TrajectoryDatasetSymbolic
    if model_type == "barcode":
        return TrajectoryDatasetBarcodes
    if model_type == "topdown":
        return TrajectoryDatasetTopdown
    raise NotImplementedError
