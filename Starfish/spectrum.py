import h5py
import numpy as np
from dataclasses import dataclass
from nptyping import NDArray
from typing import Optional
import torch


@dataclass
class Order:
    """
    A data class to hold astronomical spectra orders.

    Parameters
    ----------
    _wave : numpy.ndarray
        The full wavelength array
    _flux : numpy.ndarray
        The full flux array
    _sigma : numpy.ndarray, optional
        The full sigma array. If None, will default to all 0s. Default is None
    mask : numpy.ndarray, optional
        The full mask. If None, will default to all Trues. Default is None

    Attributes
    ----------
    name : str
    """

    _wave: torch.DoubleTensor
    _flux: torch.DoubleTensor
    _sigma: Optional[torch.DoubleTensor] = None
    mask: Optional[torch.DoubleTensor] = None

    def __post_init__(self):
        if self._sigma is None:
            self._sigma = torch.zeros_like(self._flux)
        if self.mask is None:
            self.mask = torch.ones_like(self._wave, dtype=bool)

    @property
    def wave(self):
        """
        numpy.ndarray : The masked wavelength array
        """
        return self._wave[self.mask]

    @property
    def flux(self):
        """
        numpy.ndarray : The masked flux array
        """
        return self._flux[self.mask]

    @property
    def sigma(self):
        """
        numpy.ndarray : The masked flux uncertainty array
        """
        return self._sigma[self.mask]

    def __len__(self):
        return len(self._wave)


class Spectrum:
    """
    Object to store astronomical spectra.

    Parameters
    ----------
    waves : 1D or 2D array-like
        wavelength in Angtsrom
    fluxes : 1D or 2D array-like
         flux (in f_lam)
    sigmas : 1D or 2D array-like, optional
        Poisson noise (in f_lam). If not specified, will be zeros. Default is None
    masks : 1D or 2D array-like, optional
        Mask to blot out bad pixels or emission regions. Must be castable to boolean. If None, will create a mask of all True. Default is None
    name : str, optional
        The name of this spectrum. Default is "Spectrum"

    Note
    ----
    If the waves, fluxes, and sigmas are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1 in the 0-axis.

    Warning
    -------
    For now, the Spectrum waves, fluxes, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    Attributes
    ----------
    name : str
        The name of the spectrum
    """

    def __init__(self, waves, fluxes, sigmas=None, masks=None, name="Spectrum"):
        self._waves = torch.atleast_2d(waves)
        self._fluxes = torch.atleast_2d(fluxes)

        if sigmas is not None:
            self._sigmas = torch.atleast_2d(sigmas)
        else:
            self._sigmas = torch.ones_like(fluxes)

        if masks is not None:
            self.masks = torch.atleast_2d(masks).to(bool)
        else:
            self.masks = torch.ones_like(self._waves, dtype=bool)
        assert self._fluxes.shape == self._waves.shape, "flux array incompatible shape."
        assert self._sigmas.shape == self._waves.shape, "sigma array incompatible shape."
        assert self.masks.shape == self._waves.shape, "mask array incompatible shape."

        self.name = name

    def __getitem__(self, index: int):
        return Order(self._waves[index], self._fluxes[index], self._sigmas[index], self.masks[index])

    def __setitem__(self, index: int, order: Order):
        if len(order) != self._waves.shape[1]:
            raise ValueError("Invalid order length; no ragged spectra allowed")
        self._waves[index] = order._wave
        self._fluxes[index] = order._flux
        self.masks[index] = order.mask
        self._sigmas[index] = order._sigma

    def __len__(self):
        return self._waves.shape[0]

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            n, self._n = self._n, self._n + 1
            return self[n]
        else:
            raise StopIteration

    # Masked properties
    @property
    def waves(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked wavelength arrays
        """
        return self._waves[self.masks]

    @property
    def fluxes(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked flux arrays
        """
        return self._fluxes[self.masks]

    @property
    def sigmas(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked flux uncertainty arrays
        """
        return self._sigmas[self.masks]

    @property
    def shape(self):
        """
        numpy.ndarray: The shape of the spectrum, *(norders, npixels)*

        :setter: Tries to reshape the data into a new arrangement of orders and pixels following numpy reshaping rules.
        """
        return self._waves.shape

    @shape.setter
    def shape(self, shape):
        self.reshape(shape)

    def reshape(self, shape):
        """
        Reshape the spectrum to the new shape. Obeys the same rules that numpy reshaping does. Note this is not done in-place.

        Parameters
        ----------
        shape : tuple
            The new shape of the spectrum. Must abide by numpy reshaping rules.

        Returns
        -------
        Spectrum
            The reshaped spectrum
        """
        self._waves = self._waves.reshape(shape)
        self._fluxes = self._fluxes.reshape(shape)
        self._sigmas = self._sigmas.reshape(shape)
        self.masks = self.masks.reshape(shape)

    @classmethod
    def load(cls, filename):
        """
        Load a spectrum from an hdf5 file

        Parameters
        ----------
        filename : str or path-like
            The path to the HDF5 file.

        See Also
        --------
        :meth:`save`
        """
        with h5py.File(filename, "r") as base:
            if "name" in base.attrs:
                name = base.attrs["name"]
            else:
                name = None
            waves = torch.DoubleTensor(base["waves"][:])
            fluxes = torch.DoubleTensor(base["fluxes"][:])
            sigmas = torch.DoubleTensor(base["sigmas"][:])
            masks = torch.BoolTensor(base["masks"][:])
        return cls(waves, fluxes, sigmas, masks, name=name)

    def save(self, filename):
        """
        Takes the current DataSpectrum and writes it to an HDF5 file.

        Parameters
        ----------
        filename: str or path-like
            The filename to write to. Will not create any missing directories.

        See Also
        --------
        :meth:`load`
        """

        with h5py.File(filename, "w") as base:
            base.create_dataset("waves", data=self.waves, compression=9)
            base.create_dataset("fluxes", data=self.fluxes, compression=9)
            base.create_dataset("sigmas", data=self.sigmas, compression=9)
            base.create_dataset("masks", data=self.masks, compression=9)
            if self.name is not None:
                base.attrs["name"] = self.name

    def plot(self, ax=None, **kwargs):
        """
        Plot all the orders of the spectrum

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            If provided, will plot on this axis. Otherwise, will create a new axis, by
            default None

        Returns
        -------
        matplotlib.Axes
            The axis that was plotted on
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        plot_params = {"lw": 0.5}
        plot_params.update(kwargs)
        # Plot orders
        for i, order in enumerate(self):
            ax.plot(order._wave.numpy(), order._flux.numpy(), label=f"Order: {i}", **plot_params)

        # Now plot masks
        ylims = ax.get_ylim()
        for i, order in enumerate(self):
            ax.fill_between(
                order._wave.numpy(),
                *ylims,
                color="k",
                alpha=0.1,
                where=~order.mask.numpy(),
                label="Mask" if i == 0 else None,
            )

        ax.set_yscale("log")
        ax.set_ylabel(r"$f_\lambda$ [$erg/cm^2/s/cm$]")
        ax.set_xlabel(r"$\lambda$ [$\AA$]")
        ax.legend()
        if self.name is not None:
            ax.set_title(self.name)
        fig.tight_layout()
        return ax

    def __repr__(self):
        return f"{self.name} ({len(self)} orders)"
