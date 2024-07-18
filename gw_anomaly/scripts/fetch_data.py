import os
import numpy as np
import argparse

from gwpy.timeseries import TimeSeries

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import CHANNEL


def main(args):

    segments = np.load(args.intersections)
    # dirty trick for O3b
    if len(segments)==2: segments = [segments]

    for segment in segments:

        if os.path.exists(f'{args.folder_path}/{segment[0]}_{segment[1]}/data_{args.site}.h5'):
            print(f'Already exists: {args.folder_path}/{segment[0]}_{segment[1]}/data_{args.site}.h5')
            continue
        else:
            if not os.path.exists(f'{args.folder_path}/{segment[0]}_{segment[1]}/'):
                os.mkdir(f'{args.folder_path}/{segment[0]}_{segment[1]}/')
            try:
                data = TimeSeries.get(f'{args.site}:{CHANNEL}', segment[0], segment[1])
                data.write(f'{args.folder_path}/{segment[0]}_{segment[1]}/data_{args.site}.h5')
                print(f'Fetching completed for {segment[0]} {segment[1]}')
            except:
                continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('folder_path', type=str,
                        help='Path to the Omicron output')

    parser.add_argument('intersections', type=str,
                        help='Path to the intersections file')

    parser.add_argument('--site', type=str, choices=['L1', 'H1'],
                        help='Where to save the file with injections')

    args = parser.parse_args()
    main(args)
