import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from astropy import units as u
import matplotlib.pyplot as plt
from models import LinearModel, GwakClassifier
from evaluate_data import full_evaluation
import json
import matplotlib
from config import (
    CHANNEL,
    GPU_NAME,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    VERSION,
    PERIOD
    )
from helper_functions import clean_gw_events
DEVICE = torch.device(GPU_NAME)


def whiten_bandpass_resample_clean(
        start_point,
        end_point,
        event_times_path='data/LIGO_EVENT_TIMES.npy',
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH):

    #CHANNEL = 'DCS-ANALYSIS_READY_C01:1'
    # Load LIGO data
    strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
    strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point) #.get, verbose,,, .find

    # Whiten, bandpass, and resample
    strainL1 = strainL1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
    strainL1 = strainL1.whiten()
    strainH1 = strainH1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
    strainH1 = strainH1.whiten()

    data = np.stack([strainH1, strainL1])

    event_times = np.load(event_times_path)
    # automatically chops off the first and last 10 seconds, even with no BBH
    data_cleaned = clean_gw_events(event_times,
                                   data,
                                   start_point,
                                   end_point)

    return data_cleaned


def make_eval_chunks(a, b, dur):
    '''
    Split up into one-hour chunks to normalize the whitening duration
    a, b - ints
    A, B - strings

    output - only care about the strings
    '''
    n_full_chunks = (b-a)//dur

    out = []
    for n in range(1, n_full_chunks+1):
        out.append([str(a+(n-1)*dur), str(a+n*dur)])

    # ending chunk, but still make it one hour
    out.append([str(b-dur), str(b)])

    return out


def main(args):

    save_path = f'output/{VERSION}/{args.start}_{args.end}'

    try:
        os.makedirs(save_path)
    except FileExistsError:
        None

    valid_segments = np.load(f'output/{PERIOD}_intersections.npy')

    for seg in valid_segments:
            a, b = seg
            if b-a > 3600:
                chunks = make_eval_chunks(a, b, 3600)
                for A, B in chunks:
                     A, B = int(A), int(B)

                     if args.start < A and B < args.end:

                        try:
                            data_cleaned = whiten_bandpass_resample_clean(A, B)
                            np.save(f"{save_path}/{A}_{B}.npy", np.array(data_cleaned))
                        except:
                            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('start', help="starting point to fetch 1-hour segments", type=int)
    parser.add_argument('end', help="ending point to fetch 1-hour segments", type=int)

    args = parser.parse_args()

    main(args)
