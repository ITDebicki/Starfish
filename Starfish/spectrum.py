import h5py
import numpy as np
from dataclasses import dataclass
from nptyping import NDArray
from typing import Optional, Union
import torch
import pandas as pd


@dataclass
class Order:
    """
    A data class to hold astronomical spectra orders.

    Parameters
    ----------
    _wave : torch.DoubleTensor
        The full wavelength array
    _flux : torch.DoubleTensor
        The full flux array
    _sigma : torch.DoubleTensor, optional
        The full sigma array. If None, will default to all 0s. Default is None
    mask : torch.DoubleTensor, optional
        The full mask. If None, will default to all Trues. Default is None
    device: Union[str, torch.torch.DeviceObjType], optional
        The device on which the Order is stored. If None, will default to 'cpu'.

    Attributes
    ----------
    name : str
    """

    _wave: torch.DoubleTensor
    _flux: torch.DoubleTensor
    _sigma: Optional[torch.DoubleTensor] = None
    mask: Optional[torch.DoubleTensor] = None
    device: Union[str, torch.DeviceObjType] = 'cpu'

    def __post_init__(self):
        if self._sigma is None:
            self._sigma = torch.zeros_like(self._flux)
        if self.mask is None:
            self.mask = torch.ones_like(self._wave, dtype=bool)

    def to(self, device: Union[str, torch.torch.DeviceObjType]):
        self.device = device
        self._wave = self._wave.to(device)
        self._sigma = self._sigma.to(device)
        self.mask = self.mask.to(device)
        self._flux = self._flux.to(device)

    @property
    def wave(self):
        """
        torch.DoubleTensor : The masked wavelength array
        """
        return self._wave[self.mask]

    @property
    def flux(self):
        """
        torch.DoubleTensor : The masked flux array
        """
        return self._flux[self.mask]

    @property
    def sigma(self):
        """
        torch.DoubleTensor : The masked flux uncertainty array
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

    def __init__(self, waves, fluxes, sigmas=None, masks=None, name="Spectrum", device = 'cpu'):
        
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

        self.device = device
        self._waves = self._waves.to(device)
        self._sigmas = self._sigmas.to(device)
        self.masks = self.masks.to(device)
        self._fluxes = self._fluxes.to(device)

        self.name = name

    def __getitem__(self, index: int):
        return Order(self._waves[index], self._fluxes[index], self._sigmas[index], self.masks[index], self.device)

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

    def to(self, device):
        self.device = device
        self._waves = self._waves.to(device)
        self._sigmas = self._sigmas.to(device)
        self.masks = self.masks.to(device)
        self._fluxes = self._fluxes.to(device)

    # Masked properties
    @property
    def waves(self) -> torch.DoubleTensor:
        """
        torch.DoubleTensor : The 2 dimensional masked wavelength arrays
        """
        return self._waves[self.masks]

    @property
    def fluxes(self) -> torch.DoubleTensor:
        """
        torch.DoubleTensor : The 2 dimensional masked flux arrays
        """
        return self._fluxes[self.masks]

    @property
    def sigmas(self) -> torch.DoubleTensor:
        """
        torch.DoubleTensor : The 2 dimensional masked flux uncertainty arrays
        """
        return self._sigmas[self.masks]

    @property
    def shape(self):
        """
        torch.DoubleTensor: The shape of the spectrum, *(norders, npixels)*

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
    def load(cls, filename: str, file_type: Optional[str] = None, convert = None, name = None, spectrum_filter = None, device = 'cpu', **kwargs):
        """
        Load a spectrum from a file

        Parameters
        ----------
        filename : str or path-like
            The path to the HDF5, tsv or csv file.
        fil

        See Also
        --------
        :meth:`save`
        """
        file_handlers = {
            'csv': cls.load_pd,
            'tsv': cls.load_pd,
            'hdf5': cls.load_hdf5
        }
        # Determine file type
        if file_type is None:
            file_type_mapping = {
                "txt": "csv",
                "csv": "csv",
                "tsv": "tsv",
                "hdf5": "hdf5"
            }
            parts = filename.rsplit(".", 1)
            ext = parts[-1]
            if ext.lower() in file_type_mapping:
                file_type = file_type_mapping[ext]
            else:
                raise ValueError(f"Unknown file type {ext}. Currently supported file types are: txt, csv, tsv & hdf5")
        
        if file_type not in file_handlers:
            raise ValueError(f"Unsupported file type {file_type}")
        
        waves, fluxes, sigmas, masks, name = file_handlers[file_type](filename, file_type, name, spectrum_filter = spectrum_filter, **kwargs)

        if waves is None or fluxes is None:
            raise ValueError("Waves and fluxes must be defined.")

        # Determine if need to convert units
        if convert is not None:
            # Target units are:
            # waves: Angstroms
            # fluxes: erg/cm^2/s/cm
            # sigmas: erg/cm^2/s/cm
            if isinstance(convert, dict):
                if "flux" in convert:
                    fluxes = convert['flux'](fluxes)
                    if sigmas is not None:
                        sigmas = convert['flux'](sigmas)
                
                if 'waves' in convert:
                    waves = convert['waves'](waves)
            else:
                if len(convert) == 1:
                    waves = convert[0](waves)
                elif len(convert) >= 2:
                    if convert[0] is not None:
                        waves = convert[0](waves)

                    if convert[1] is not None:
                        fluxes = convert[1](fluxes)
                        if sigmas is not None:
                            sigmas = convert[1](sigmas)


        return cls(torch.tensor(waves, dtype = torch.float64),
                    torch.tensor(fluxes, dtype = torch.float64),
                    torch.tensor(sigmas, dtype = torch.float64),
                    torch.tensor(masks, dtype = torch.bool),
                    name=name, device = device)

    @staticmethod
    def load_pd(filename, file_type, name, spectrum_filter = None, **kwargs):
        df = pd.read_csv(filename, delimiter='\t' if file_type == 'tsv' else ',', **kwargs)
        df = df.sort_values('waves')
        df = spectrum_filter(df) if spectrum_filter else df
        waves = df["waves"].values if "waves" in df else None
        fluxes = df["fluxes"].values if "fluxes" in df else None
        sigmas = df["sigmas"].values if "sigmas" in df else None
        masks = df["masks"].values if "masks" in df else None

        if masks is None and fluxes is not None:
            masks = ~np.isnan(fluxes)
            if sigmas is not None:
                masks = masks & ~np.isnan(sigmas)

        return waves, fluxes, sigmas, masks, name


    @staticmethod
    def load_hdf5(filename: str, file_type, name, spectrum_filter = None, **kwargs):
        with h5py.File(filename, "r") as base:
            if name is None and "name" in base.attrs:
                name = base.attrs["name"]
            waves = base["waves"][:]
            fluxes = base["fluxes"][:]
            sigmas = base["sigmas"][:]
            masks = base["masks"][:]
        
        return waves, fluxes, sigmas, masks, name

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