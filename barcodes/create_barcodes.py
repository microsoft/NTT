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
from PIL import Image
import numpy as np
import itertools
import base64
import io
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Takes Bleeding Edge JSON replays and creates "barcodes".',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--indir',
        type=str,
        help='Path to folders with the JSON to convert to barcodes.')
    parser.add_argument('--outdir', type=str, default="barcode_data",
                        help='Output directory.')

    args = parser.parse_args()

    IN_DIR = args.indir
    OUT_DIR = args.outdir

    os.makedirs(OUT_DIR, exist_ok=True)
    list_in_dirs = os.listdir(IN_DIR)
    for f, filename in enumerate(list_in_dirs):
        file = os.path.join(IN_DIR, filename)
        if filename == 'sets.json':
            print(f"{f+1}/{len(list_in_dirs)}: Skipping sets.json")
            continue
        video = []
        with open(file) as main_file:
            for line in itertools.islice(main_file, 0, None, 1):
                step = json.loads(line)
                key = list(step.keys())[0]
                encoded_img = step[key]["Observations"]["Players"][0]["Image"]["ImageBytes"]
                decoded_image_data = base64.decodebytes(
                    encoded_img.encode('utf-8'))
                image = Image.open(io.BytesIO(decoded_image_data))
                img = np.array(image)
                video.append(img)

        # compute barcodes
        videodata = np.array(video)
        size = videodata.shape
        barcode = np.zeros((size[0], size[2], 3))

        for t in range(0, size[0]):
            frame = videodata[t, :]
            x_sum = np.sum(frame, axis=0)
            barcode[t] = x_sum / size[1]

        print(
            f"{f+1}/{len(list_in_dirs)}: Creating barcode with shape",
            barcode.shape)
        img = Image.fromarray(barcode.astype(np.uint8))
        png_filename = os.path.splitext(filename)[0] + '.png'
        img.save(os.path.join(OUT_DIR, png_filename))
