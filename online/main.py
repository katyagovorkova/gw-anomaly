import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from ligo.gracedb.rest import GraceDb
from utils.buffer import InputBuffer, OutputBuffer
from utils.dataloading import data_iterator
from utils.gdb import gracedb_factory
from utils.searcher import Event, Searcher
from utils.snapshotter import OnlineSnapshotter

from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

# seconds of data per update
UPDATE_SIZE = 1


def load_model(model: Architecture, weights: Path):
    checkpoint = torch.load(weights, map_location="cpu")
    arch_weights = {
        k: v for k, v in checkpoint.items() if k.startswith("model.")
    }
    model.load_state_dict(arch_weights)
    model.to("cuda")
    model.eval()
    return model, checkpoint


def get_time_offset(
    inference_sampling_rate: float,
    fduration: float,
    integration_window_length: float,
    kernel_length: float,
    trigger_distance: float,
):
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

    return time_offset


def search(
    gdb: GraceDb,
    pe_whitener: Whiten,
    scaler: torch.nn.Module,
    spectral_density: SpectralDensity,
    whitener: BatchWhitener,
    snapshotter: OnlineSnapshotter,
    searcher: Searcher,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    gwak: Architecture,
    amplfi: Architecture,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    time_offset: float,
    outdir: Path,
):
    integrated = None

    # flat that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    #
    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        X = X.to("cuda")

        # if this frame was not analysis ready
        if not ready:
            if searcher.detecting:
                # if we were in the middle of a detection,
                # we won't get to see the peak of the event
                # so build the event with what we have
                event = searcher.build_event(
                    integrated, t0 - 1, len(integrated) - 1
                )
                if event is not None:
                    # maybe process event found in the previous frame
                    gdb.submit(event)
                    archer.detecting = False

            # check if this is because the frame stream stopped
            # being analysis ready, in which case perform updates
            # but don't search for events
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring any triggers".format(t0)
                )
            # or if it's because frames were dropped within the stream
            # in which case we should reset our states
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )

                input_buffer.reset()
                output_buffer.reset()

                # nothing left to do, so move on to next frame
                continue

        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            state = snapshotter.reset()
            input_buffer.reset()
            output_buffer.reset()
            in_spec = True

        # we have a frame that is analysis ready,
        # so lets analyze it:

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X, state)

        # whiten the batch, and analyze with gwak
        whitened = whitener(batch)
        y = gwak(whitened)[:, 0]

        # update our input buffer with latest strain data,
        input_buffer.update(X)
        # update our output buffer with the latest gwak output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y, t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        if snapshotter.full_psd_present and ready:
            event = searcher.search(integrated, t0 + time_offset)

        # if we found an event, process it!
        if event is not None:
            gdb.submit(event)
            searcher.detecting = False

        # TODO write buffers to disk:


def main(
    gwak_architecture: Architecture,
    gwak_weights: Path,
    amplfi_architecture: Architecture,
    amplfi_weights: Path,
    background_file: Path,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    inference_params: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    trigger_distance: float,
    fduration: float,
    integration_window_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: str = "test",
    ifo_suffix: str = None,
    input_buffer_length=75,
    output_buffer_length=8,
):
    gdb = gracedb_factory(server)
    num_ifos = len(ifos)

    # initialize a buffer for storing recent strain data,
    # and for storing integrated gwak outputs
    input_buffer = InputBuffer(
        num_channels=num_ifos,
        sample_rate=sample_rate,
        buffer_length=input_buffer_length,
        fduration=fduration,
    )
    output_buffer = OutputBuffer(
        inference_sampling_rate=inference_sampling_rate,
        integration_window_length=integration_window_length,
        buffer_length=output_buffer_length,
    )


    print('I AM HERE')

    # # Load in gwak and amplfi models
    # logging.info(
    #     f"Loading GWAK from weights at path {gwak_weights}\n"
    #     f"Loading AMPLFI from weights at path {amplfi_weights}"
    # )
    # gwak, _ = load_model(gwak_architecture, gwak_weights)
    # amplfi, scaler = load_amplfi(
    #     amplfi_architecture, amplfi_weights, len(inference_params)
    # )

    # fftlength = fftlength or kernel_length + fduration

    # whitener = BatchWhitener(
    #     kernel_length=kernel_length,
    #     sample_rate=sample_rate,
    #     inference_sampling_rate=inference_sampling_rate,
    #     batch_size=UPDATE_SIZE * inference_sampling_rate,
    #     fduration=fduration,
    #     fftlength=fftlength,
    #     highpass=highpass,
    # )

    # snapshotter = OnlineSnapshotter(
    #     psd_length=psd_length,
    #     kernel_length=kernel_length,
    #     fduration=fduration,
    #     sample_rate=sample_rate,
    #     inference_sampling_rate=inference_sampling_rate,
    # )


    time_offset = get_time_offset(
        inference_sampling_rate,
        fduration,
        integration_window_length,
        kernel_length,
        trigger_distance,
    )

    data_it = data_iterator(
        datadir=datadir,
        channel=channel,
        ifos=ifos,
        sample_rate=sample_rate,
        ifo_suffix=ifo_suffix,
        timeout=10,
    )


    print('Got some data!! ')
    print(data_it)

    search(
        gdb,
        pe_whitener,
        scaler,
        spectral_density,
        whitener,
        snapshotter,
        searcher,
        input_buffer,
        output_buffer,
        gwak,
        amplfi,
        data_it,
        outdir,
        time_offset,
    )


if __name__ == "__main__":
    main()