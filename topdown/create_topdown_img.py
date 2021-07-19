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
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from symbolic.symbolic_dataset import read_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Takes Bleeding Edge JSON replays and creates images of the top-down trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--background',
        type=str,
        default="black",
        choices=[
            "white",
            "black"],
        help='Background color of the image (black or white).')
    parser.add_argument(
        '--resolution',
        type=int,
        default=[
            200,
            320],
        nargs='+',
        help='Image resolution.')
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        help='Path to folders with the JSON to convert to images.')
    parser.add_argument(
        '--outdir',
        type=str,
        default=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/td_data",
        help='Output directory.')
    parser.add_argument(
        '--border',
        type=int,
        default=0,
        help='Size of the image border, useful for visuals, not models.')

    args = parser.parse_args()

    im_res = args.resolution
    folders_to_convert = args.folders

    assert len(im_res) == 2
    assert len(folders_to_convert) > 0

    color = 255 if args.background == "black" else 0

    i = 0
    for folder in folders_to_convert:
        target_folder = os.path.join(
            args.outdir, os.path.basename(
                os.path.normpath(folder)))
        os.makedirs(target_folder, exist_ok=True)

        all_trajectories, ranges = read_trajectories([folder], "unknown")
        for traj_data in all_trajectories:
            if args.background == "white":
                image = np.ones(im_res, dtype=np.uint8) * 255
            elif args.background == "black":
                image = np.zeros(im_res, dtype=np.uint8)
            else:
                raise NotImplementedError(
                    "Only supported background colors are black and white.")

            for pos in traj_data["obs"]:
                arr_x = min(int(pos[0] * im_res[0]), im_res[0] - 1)
                arr_y = min(int(pos[1] * im_res[1]), im_res[1] - 1)
                image[arr_x, arr_y] = color
            for t in range(args.border):
                image[:, t] = color
                image[t, :] = color
                image[im_res[0] - 1 - t, :] = color
                image[:, im_res[1] - 1 - t] = color
            im = Image.fromarray(image)
            target_file = os.path.join(target_folder, "td_" + str(i) + ".png")
            i += 1
            im.save(target_file)
