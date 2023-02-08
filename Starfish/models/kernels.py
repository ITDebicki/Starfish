import numpy as np
from typing import List

import torch
from Starfish import constants as C


def global_covariance_matrix(
    wave: torch.Tensor, amplitude: float, lengthscale: float
) -> torch.Tensor:
    """
    A matern-3/2 kernel where the metric is defined as the velocity separation of the wavelengths.

    Parameters
    ----------
    wave : torch.Tensor
        The wavelength grid
    amplitude : float
        The amplitude of the kernel
    lengthscale : float
        The lengthscale of the kernel

    Returns
    -------
    cov : torch.Tensor
        The global covariance matrix
    """
    wx, wy = torch.meshgrid(wave, wave, indexing='xy')
    r = C.c_kms / 2 * torch.abs((wx - wy) / (wx + wy))
    r0 = 6 * lengthscale

    # Calculate the kernel, being careful to stay in mask
    kernel = torch.zeros((len(wx), len(wy)), dtype = wave.dtype, device = wave.device)
    mask = r <= r0
    taper = 0.5 + 0.5 * torch.cos(torch.pi * r[mask] / r0)
    kernel[mask] = (
        taper
        * amplitude
        * (1 + np.sqrt(3) * r[mask] / lengthscale)
        * torch.exp(-np.sqrt(3) * r[mask] / lengthscale)
    )
    return kernel


def local_covariance_matrix(
    wave: torch.Tensor, amplitude: float, mu: float, sigma: float
) -> torch.Tensor:
    """
    A local Gaussian kernel. In general, the kernel has density like

    .. math::
        K(\\lambda | A, \\mu, \\sigma) = A \\exp\\left[-\\frac12 \\frac{\\left(\\lambda - \\mu\\right)^2}{\\sigma^2} \\right]

    Parameters
    ----------
    wave : torch.Tensor
        The wavelength grid
    amplitude : float
        The amplitudes of the Gaussian
    mu : float
        The means of the Gaussian
    sigma : float
        The standard deviations of the Gaussian

    Returns
    -------
    cov : torch.Tensor
        The sum of each Gaussian kernel, or the local covariance kernel
    """
    # Set up the metric and mesh grid
    met = C.c_kms / mu * torch.abs(wave - mu)
    x, y = torch.meshgrid(met, met, indexing='xy')
    r_tap, _ = torch.max(torch.stack([x, y]), axis=0)
    r2 = x**2 + y**2
    r0 = 4 * sigma

    # Calculate the kernel. Use masking to keep sparse-ish calculations
    kernel = torch.zeros((len(x), len(y)), dtype = wave.dtype, device = wave.device)
    mask = r_tap <= r0
    taper = 0.5 + 0.5 * torch.cos(torch.pi * r_tap[mask] / r0)
    kernel[mask] = taper * amplitude * torch.exp(-0.5 * r2[mask] / sigma**2)
    return kernel
