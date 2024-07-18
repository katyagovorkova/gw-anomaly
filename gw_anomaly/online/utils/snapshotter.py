from typing import Callable, Optional, Tuple

import torch

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows

from .buffer import GPU

Tensor = torch.Tensor

class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    two unequal-length segments, calculates the PSD of
    the first section, then returns this PSD along with
    the second section.

    Args:
        length:
            The length, in seconds, of timeseries data
            to be returned for whitening. Note that the
            length of time used for the PSD will then be
            whatever remains along first part of the time
            axis of the input.
        sample_rate:
            Rate at which input data has been sampled in Hz
        fftlength:
            Length of FFTs to use when computing the PSD
        overlap:
            Amount of overlap between FFT windows when
            computing the PSD. Default value of `None`
            uses `fftlength / 2`
        average:
            Method for aggregating spectra from FFT
            windows, either `"mean"` or `"median"`
        fast:
            If `True`, use a slightly faster PSD algorithm
            that is inaccurate for the lowest two frequency
            bins. If you plan on highpassing later, this
            should be fine.
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "median",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, fast=fast
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)

        # if we have 2 batch elements in our input data,
        # it will be assumed that the 0th element is data
        # being used to calculate the psd to whiten the
        # 1st element. Used when we want to use raw background
        # data to calculate the PSDs to whiten data with injected signals
        if X.ndim == 3 and X.size(0) == 2:
            # 0th background element is used to calculate PSDs
            background = background[0]
            # 1st element is the data to be whitened
            X = X[1]

        psds = self.spectral_density(background.double())
        return X, psds


class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(self, update: Tensor, snapshot: Tensor) -> Tuple[Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        augmentor: Optional[Callable] = None,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate) # must be 200 --> kernel_length = 200/sample_rate
        self.augmentor = augmentor

        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)

    def forward(self, x: Tensor) -> Tensor:
        # Get the number of channels so we know how to
        # reshape `x` appropriately after unfolding to
        # ensure we have (batch, channels, time) shape
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        x, psd = self.psd_estimator(x)
        x = self.whitener(x.double(), psd)

        # unfold x and then put it into the expected shape.
        # Note that if x has both signal and background
        # batch elements, they will be interleaved along
        # the batch dimension after unfolding
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        x = x.reshape(-1, num_channels, self.kernel_size)
        if self.augmentor is not None:
            x = self.augmentor(x)
        return x


class SnapshotWhitener(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        psd_length: float,
        kernel_length: float,
        fduration: float,
        sample_rate: float,
        inference_sampling_rate: float,
        fftlength: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to(f"cuda:{GPU}")

        # Updates come in 1 second chunks, so each
        # update will generate a batch of
        # `inference_sampling_rate` overlapping
        # windows to whiten
        batch_size = 1 * inference_sampling_rate
        self.batch_whitener = BatchWhitener(
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
        ).to(f"cuda:{GPU}")

        self.step_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        self.filter_size = int(fduration * sample_rate)

        self.sample_rate = sample_rate
        self.contiguous_update_size = 0

    @property
    def state_size(self):
        return (
            self.psd_size
            + self.kernel_size
            + self.filter_size
            - self.step_size
        )

    def get_initial_state(self):
        self.contiguous_update_size = 0
        return torch.zeros((1, self.num_channels, self.state_size))

    def forward(self, update, current_state):
        update = update[None, :, :]
        print(223, update.shape, current_state.shape)
        X, current_state = self.snapshotter(update, current_state)
        print(224, update.shape, X.shape)
        # If we haven't had enough updates in a row to
        # meaningfully whiten, note that for upstream processes
        full_psd_present = (
            self.contiguous_update_size >= self.state_size - update.shape[-1]
        )
        if not full_psd_present:
            self.contiguous_update_size += update.shape[-1]
        print(231, "snapshotter", X.shape)
        return self.batch_whitener(X), current_state, full_psd_present