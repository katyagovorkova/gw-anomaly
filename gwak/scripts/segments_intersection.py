import json
import argparse
import numpy as np


def intersect(seg1, seg2):
    a, b = seg1
    c, d = seg2
    start, end = max(a, c), min(b, d)

    if start < end:
        return [start, end]

    return None


def find_intersections(period):

    hanford = json.load(open(f'data/{period}_Hanford_segments.json'))['segments']
    hanford = np.array(hanford)
    livingston = json.load(open(f'data/{period}_Livingston_segments.json'))['segments']
    livingston = np.array(livingston)

    # there aren't that many segments, so N^2 isn't so bad
    valid_segments = []
    for h_elem in hanford:
        for l_elem in livingston:
            intersection = intersect(h_elem, l_elem)
            if intersection is not None:
                valid_segments.append(intersection)

    return np.array(valid_segments)


def main(save_path, period):
    '''
    Function which takes the valid segments from both detectors
    and finds an "intersection", i.e. segments where both detectors
    are recording data

    paths are string which point to the corresponding .json files
    '''
    if period == 'O3':
        valid_segments_a = find_intersections('O3a')
        valid_segments_b = find_intersections('O3b')
        valid_segments = np.concatenate([valid_segments_a, valid_segments_b])
    else:
        valid_segments = find_intersections(period)

    np.save(save_path, np.array(valid_segments))


main(snakemake.output[0], snakemake.wildcards['period'])
