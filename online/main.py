import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from utils.buffer import DataBuffer, GPU
from utils.dataloading import data_iterator
from utils.snapshotter import SnapshotWhitener
from utils.trigger import Searcher, Trigger

from ml4gw.transforms import SpectralDensity, Whiten

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts.models import LSTM_AE_SPLIT as architecture

@torch.no_grad()
def main(
    architecture: Callable = architecture,
    outdir: Path = '.',
    weights_path: Path = '/home/katya.govorkova/gw_anomaly/output/O3av2/trained/models/bbh.pt',
    datadir: Path = '/dev/shm/kafka',
    ifos: List[str] = ['H1', 'L1'],
    channel: str = 'GDS-CALIB_STRAIN_CLEAN',
    sample_rate: float = 4096,
    kernel_length: float = 200/4096,
    inference_sampling_rate: float = 4096,
    psd_length: float = 100,
    trigger_distance: float = 1000,
    fduration: float = 1,
    integration_window_length: float = 5,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_per_day: float = 1,
    secondary_far_threshold: float = 24,
    server: str = "test",
    ifo_suffix: str = None,
    input_buffer_length=5,
    output_buffer_length=8,
    verbose: bool = False,
):

    num_ifos = len(ifos)
    buffer = DataBuffer(
        num_ifos,
        sample_rate,
        inference_sampling_rate,
        integration_window_length,
        input_buffer_length,
        output_buffer_length,
    )

    # instantiate network and load in its optimized parameters
    print(f"Build network and loading weights from {weights_path}")

    # gwak setup
    gwak = architecture(num_ifos, 200, 4).to(f'cuda:{GPU}')
    fftlength = fftlength or kernel_length + fduration
    whitener = SnapshotWhitener(
        num_channels=num_ifos,
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        fftlength=fftlength,
        highpass=highpass,
    )
    current_state = whitener.get_initial_state().to(f'cuda:{GPU}')

    weights = torch.load(weights_path)
    gwak.load_state_dict(weights)
    gwak.eval()

    # Amplfi setup. Hard code most of it for now
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(f'cuda:{GPU}')
    pe_whitener = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=highpass
    ).to(f'cuda:{GPU}')
    # amplfi, std_scaler = set_up_amplfi()

    # set up some objects to use for finding
    # and submitting triggers
    fars = [far_per_day, secondary_far_threshold]
    searcher = Searcher(
        outdir, fars, inference_sampling_rate, refractory_period
    )

    triggers = [
        Trigger(server=server, write_dir=f"{outdir}/{server}/triggers"),
        Trigger(
            server=server, write_dir=f"{outdir}/{server}/secondary-triggers"
        ),
    ]
    in_spec = True

    def get_trigger(event):
        fars_hz = [i / 3600 / 24 for i in fars]
        idx = np.digitize(event.far, fars_hz)
        if idx == 0 and not in_spec:
            logging.warning(
                "Not submitting event {} to production trigger "
                "because data is not analysis ready".format(event)
            )
            idx = 1
        return triggers[idx]

    # offset the initial timestamp of our
    # integrated outputs relative to the
    # initial timestamp of the most recently
    # loaded frames
    time_offset = (
        1 / inference_sampling_rate  # end of the first kernel in batch
        - fduration / 2  # account for whitening padding
        - integration_window_length  # account for time to build peak
    )

    if trigger_distance is not None:
        if trigger_distance > 0:
            time_offset -= kernel_length - trigger_distance
        if trigger_distance < 0:
            # Trigger distance parameter accounts for fduration already
            time_offset -= np.abs(trigger_distance) - fduration / 2

    logging.info("Beginning search")
    data_it = data_iterator(
        datadir=datadir,
        channel=channel,
        ifos=ifos,
        sample_rate=sample_rate,
        ifo_suffix=ifo_suffix,
        timeout=10,
    )
    integrated = None  # need this for static linters
    last_event_written = True
    last_event_time = 0
    for X, t0, ready in data_it:
        # adjust t0 to represent the timestamp of the
        # leading edge of the input to the network
        if not ready:
            in_spec = False

            # if we had an event in the last frame, we
            # won't get to see its peak, so do our best
            # to build the event with what we have
            if searcher.detecting and not searcher.check_refractory:
                event = searcher.build_event(
                    integrated[-1], t0 - 1, len(integrated) - 1
                )
                trigger = get_trigger(event)
                response = trigger.submit(event, ifos, datadir, ifo_suffix)
                logging.info(response.json().keys())
                searcher.detecting = False
                last_event_written = False
                last_event_trigger = trigger
                last_event_time = event.gpstime
                # bilby_res, mollview_plot = run_amplfi(
                #     last_event_time,
                #     buffer.input_buffer,
                #     fduration,
                #     spectral_density,
                #     pe_whitener,
                #     amplfi,
                #     std_scaler,
                #     outdir / "whitened_data_plots",
                # )
                # graceid = response.json()["graceid"]
                # trigger.submit_pe(bilby_res, mollview_plot, graceid)

            # check if this is because the frame stream stopped
            # being analysis ready, or if it's because frames
            # were dropped within the stream
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring any triggers".format(t0)
                )
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )
                # Write whatever data we have from the event
                if not last_event_written:
                    write_path = last_event_trigger.write_dir
                    buffer.write(write_path, last_event_time)
                    last_event_written = True
                buffer.reset_state()
                continue
        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            current_state = whitener.get_initial_state().to(f'cuda:{GPU}')
            buffer.reset_state()
            in_spec = True

        X = X.to(f'cuda:{GPU}')
        batch, current_state, full_psd_present = whitener(X, current_state)
        print(f'Batch shape is {batch.shape}')
        y = gwak(batch)[:,0,0]
        print(f'OUTPUT shape is {y.shape}')
        integrated = buffer.update(
            input_update=X,
            output_update=y,
            t0=t0,
            input_time_offset=0,
            output_time_offset=time_offset + integration_window_length,
        )

        event = None
        # Only search if we had sufficient data to whiten with
        # and if frames were analysis ready
        if full_psd_present and ready:
            event = searcher.search(integrated, t0 + time_offset)

        if event is not None:
            trigger = get_trigger(event)
            response = trigger.submit(event, ifos, datadir, ifo_suffix)
            last_event_written = False
            last_event_trigger = trigger
            last_event_time = event.gpstime
            # bilby_res, mollview_plot = run_amplfi(
            #     last_event_time,
            #     buffer.input_buffer,
            #     fduration,
            #     spectral_density,
            #     pe_whitener,
            #     amplfi,
            #     std_scaler,
            #     outdir / "whitened_data_plots",
            # )
            # graceid = response.json()["graceid"]
            # trigger.submit_pe(bilby_res, mollview_plot, graceid)

        if (
            not last_event_written
            and last_event_time + output_buffer_length / 2 < t0
        ):
            write_path = last_event_trigger.write_dir
            buffer.write(write_path, last_event_time)
            last_event_written = True


if __name__=='__main__':
    main()