from typing import Callable, Optional, Tuple

import torch

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows

Tensor = torch.Tensor


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


class OnlineSnapshotter(BackgroundSnapshotter):
    """
    Light subclass of BackgroundSnapshotter that
    registers the initial state as a buffer, and
    keeps track of contiguous update size to determine
    if there is enough data to calculate a PSD
    """

    def __init__(self, *args, update_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_size = update_size
        self.contiguous_update_size = 0
        self.register_buffer(
            "initial_state", torch.zeros((1, self.state_size))
        )

    @property
    def full_psd_present(self):
        return (
            self.contiguous_update_size >= self.state_size - self.update_size
        )

    def reset(self):
        self.contiguous_update_size = 0
        return self.initial_state

    def forward(self, update, state):
        X, state = super().forward(update, state)
        if not self.full_psd_present:
            self.contiguous_update_size += self.update.shape[-1]
        return X, state