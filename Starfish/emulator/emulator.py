import logging
import os
import warnings
from typing import Sequence, Optional, Union, Tuple

import h5py
import numpy as np
from nptyping import NDArray
from scipy.interpolate import LinearNDInterpolator
from sklearn.decomposition import PCA

from Starfish.grid_tools import HDF5Interface
from Starfish.grid_tools.utils import determine_chunk_log
from Starfish.utils import calculate_dv
from Starfish.param_dict import GroupedParamDict
from .kernels import batch_kernel
from ._utils import get_phi_squared, get_w_hat, reshape_fortran
import torch

log = logging.getLogger(__name__)


class Emulator:
    """
    A Bayesian spectral emulator.

    This emulator offers an interface to spectral libraries that offers interpolation
    while providing a variance-covariance matrix that can be forward-propagated in
    likelihood calculations. For more details, see the appendix from Czekala et al.
    (2015).

    Parameters
    ----------
    grid_points : numpy.ndarray
        The parameter space from the library.
    param_names : array-like of str
        The names of each parameter from the grid
    wavelength : numpy.ndarray
        The wavelength of the library models
    weights : numpy.ndarray
        The PCA weights for the original grid points
    eigenspectra : numpy.ndarray
        The PCA components from the decomposition
    w_hat : numpy.ndarray
        The best-fit weights estimator
    flux_mean : numpy.ndarray
        The mean flux spectrum
    flux_std : numpy.ndarray
        The standard deviation flux spectrum
    lambda_xi : float, optional
        The scaling parameter for the augmented covariance calculations, default is 1
    variances : numpy.ndarray, optional
        The variance parameters for each of Gaussian process, default is array of 1s
    lengthscales : numpy.ndarray, optional
        The lengthscales for each Gaussian process, each row should have length equal
        to number of library parameters, default is arrays of 3 * the max grid
        separation for the grid_points
    name : str, optional
        If provided, will give a name to the emulator; useful for keeping track of
        filenames. Default is None.


    Attributes
    ----------
    params : dict
        The underlying hyperparameter dictionary
    """

    def __init__(
        self,
        grid_points: NDArray[float],
        param_names: Sequence[str],
        wavelength: NDArray[float],
        weights: NDArray[float],
        eigenspectra: NDArray[float],
        w_hat: NDArray[float],
        flux_mean: NDArray[float],
        flux_std: NDArray[float],
        factors: NDArray[float],
        lambda_xi: float = 1.0,
        variances: Optional[NDArray[float]] = None,
        lengthscales: Optional[NDArray[float]] = None,
        name: Optional[str] = None,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.grid_points = torch.DoubleTensor(grid_points)
        self.param_names = param_names
        self.wl = torch.DoubleTensor(wavelength)
        self.weights = torch.DoubleTensor(weights)
        self.eigenspectra = torch.DoubleTensor(eigenspectra)
        self.flux_mean = torch.DoubleTensor(flux_mean)
        self.flux_std = torch.DoubleTensor(flux_std)
        self.factors = torch.DoubleTensor(factors)
        self.factor_interpolator = LinearNDInterpolator(
            grid_points, factors, rescale=True
        )

        
        self.dv = calculate_dv(self.wl)
        self.ncomps = eigenspectra.shape[0]

        self.hyperparams = GroupedParamDict(groupTensors=[False, True], max_depth=1)
        self.name = name

        self.lambda_xi = lambda_xi

        self.variances = torch.DoubleTensor(
            variances if variances is not None else 1e4 * np.ones(self.ncomps)
        )

        unique = [sorted(np.unique(param_set)) for param_set in self.grid_points.T]
        self._grid_sep = np.array([np.diff(param).max() for param in unique])

        if lengthscales is None:
            lengthscales = np.tile(3 * self._grid_sep, (self.ncomps, 1))

        self.lengthscales = torch.DoubleTensor(lengthscales)

        # Determine the minimum and maximum bounds of the grid
        self.min_params = self.grid_points.min(axis=0).values
        self.max_params = self.grid_points.max(axis=0).values

        # TODO find better variable names for the following
        self.iPhiPhi = torch.DoubleTensor(torch.linalg.inv(
            get_phi_squared(self.eigenspectra, self.grid_points.shape[0])
        ))
        self.v11 = self.iPhiPhi / self.lambda_xi + batch_kernel(
            self.grid_points, self.grid_points, self.variances, self.lengthscales
        )
        self.w_hat = torch.DoubleTensor(w_hat)

        self._trained = False

    @property
    def lambda_xi(self) -> float:
        """
        float : The tuning hyperparameter

        :setter: Sets the value.
        """
        return torch.exp(self.hyperparams["log_lambda_xi"])

    @lambda_xi.setter
    def lambda_xi(self, value: float):
        self.hyperparams["log_lambda_xi"] = np.log(value)

    @property
    def variances(self) -> NDArray[float]:
        """
        numpy.ndarray : The variances for each Gaussian process kernel.

        :setter: Sets the variances given an array.
        """
        return torch.exp(self.hyperparams['log_variance'].values()[0])

    @variances.setter
    def variances(self, values: NDArray[float]):
        for i, value in enumerate(values):
            self.hyperparams[f"log_variance:{i}"] = np.log(value)

    @property
    def lengthscales(self) -> NDArray[float]:
        """
        numpy.ndarray : The lengthscales for each Gaussian process kernel.

        :setter: Sets the lengthscales given a 2d array
        """
        return torch.exp(self.hyperparams['log_lengthscale'].values()[0]).reshape(self.ncomps, -1)

    @lengthscales.setter
    def lengthscales(self, values: NDArray[float]):
        for i, value in enumerate(values):
            for j, ls in enumerate(value):
                self.hyperparams[f"log_lengthscale:{i}:{j}"] = np.log(ls)

    def __getitem__(self, key):
        return self.hyperparams[key]

    @classmethod
    def load(cls, filename: Union[str, os.PathLike]):
        """
        Load an emulator from and HDF5 file

        Parameters
        ----------
        filename : str or path-like
        """
        filename = os.path.expandvars(filename)
        with h5py.File(filename, "r") as base:
            grid_points = base["grid_points"][:]
            param_names = base["grid_points"].attrs["names"]
            wavelength = base["wavelength"][:]
            weights = base["weights"][:]
            eigenspectra = base["eigenspectra"][:]
            flux_mean = base["flux_mean"][:]
            flux_std = base["flux_std"][:]
            w_hat = base["w_hat"][:]
            factors = base["factors"][:]
            lambda_xi = base["hyperparameters"]["lambda_xi"][()]
            variances = base["hyperparameters"]["variances"][:]
            lengthscales = base["hyperparameters"]["lengthscales"][:]
            trained = base.attrs["trained"]
            if "name" in base.attrs:
                name = base.attrs["name"]
            else:
                name = ".".join(filename.split(".")[:-1])

        emulator = cls(
            grid_points=grid_points,
            param_names=param_names,
            wavelength=wavelength,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std,
            lambda_xi=lambda_xi,
            variances=variances,
            lengthscales=lengthscales,
            name=name,
            factors=factors,
        )
        emulator._trained = trained
        return emulator

    def save(self, filename: Union[str, os.PathLike]):
        """
        Save the emulator to an HDF5 file

        Parameters
        ----------
        filename : str or path-like
        """
        filename = os.path.expandvars(filename)
        with h5py.File(filename, "w") as base:
            grid_points = base.create_dataset(
                "grid_points", data=self.grid_points, compression=9
            )
            grid_points.attrs["names"] = self.param_names
            waves = base.create_dataset("wavelength", data=self.wl, compression=9)
            waves.attrs["units"] = "Angstrom"
            base.create_dataset("weights", data=self.weights, compression=9)
            eigens = base.create_dataset(
                "eigenspectra", data=self.eigenspectra, compression=9
            )
            base.create_dataset("flux_mean", data=self.flux_mean, compression=9)
            base.create_dataset("flux_std", data=self.flux_std, compression=9)
            eigens.attrs["units"] = "erg/cm^2/s/Angstrom"
            base.create_dataset("w_hat", data=self.w_hat, compression=9)
            base.attrs["trained"] = self._trained
            if self.name is not None:
                base.attrs["name"] = self.name
            base.create_dataset("factors", data=self.factors, compression=9)
            hp_group = base.create_group("hyperparameters")
            hp_group.create_dataset("lambda_xi", data=self.lambda_xi)
            hp_group.create_dataset("variances", data=self.variances, compression=9)
            hp_group.create_dataset(
                "lengthscales", data=self.lengthscales, compression=9
            )

        self.log.info("Saved file at {}".format(filename))

    @classmethod
    def from_grid(cls, grid, **pca_kwargs):
        """
        Create an Emulator using PCA decomposition from a GridInterface.

        Parameters
        ----------
        grid : :class:`GridInterface` or str
            The grid interface to decompose
        pca_kwargs : dict, optional
            The keyword arguments to pass to PCA. By default, `n_components=0.99` and
            `svd_solver='full'`.

        See Also
        --------
        sklearn.decomposition.PCA
        """
        # Load grid if a string is given
        if isinstance(grid, str):
            grid = HDF5Interface(grid)

        fluxes = np.array(list(grid.fluxes))
        # Normalize to an average of 1 to remove uninteresting correlation
        norm_factors = fluxes.mean(1)
        fluxes /= norm_factors[:, np.newaxis]
        # Center and whiten
        flux_mean = fluxes.mean(0)
        fluxes -= flux_mean
        flux_std = fluxes.std(0)
        fluxes /= flux_std

        # Perform PCA using sklearn
        default_pca_kwargs = dict(n_components=0.99, svd_solver="full")
        default_pca_kwargs.update(pca_kwargs)
        pca = PCA(**default_pca_kwargs)
        weights = pca.fit_transform(fluxes)
        eigenspectra = pca.components_

        exp_var = pca.explained_variance_ratio_.sum()
        # This is basically the mean square error of the reconstruction
        log.info(
            f"PCA fit {exp_var * 100:.2f}% of the variance with {pca.n_components_:d} components."
        )


        w_hat = get_w_hat(torch.DoubleTensor(eigenspectra), torch.DoubleTensor(fluxes))

        emulator = cls(
            grid_points=grid.grid_points,
            param_names=grid.param_names,
            wavelength=grid.wl,
            weights=weights,
            eigenspectra=eigenspectra,
            w_hat=w_hat,
            flux_mean=flux_mean,
            flux_std=flux_std,
            factors=norm_factors,
        )
        return emulator

    def __call__(
        self,
        params: Sequence[float],
        full_cov: bool = True,
        reinterpret_batch: bool = False,
    ) -> Tuple[NDArray[float], NDArray[float]]:
        """
        Gets the mu and cov matrix for a given set of params

        Parameters
        ----------
        params : array_like
            The parameters to sample at. Should be consistent with the shapes of the
            original grid points.
        full_cov : bool, optional
            Return the full covariance or just the variance, default is True. This will
            have no effect of reinterpret_batch is true
        reinterpret_batch : bool, optional
            Will try and return a batch of output matrices if the input params are a
            list of params, default is False.

        Returns
        -------
        mu : numpy.ndarray (len(params),)
        cov : numpy.ndarray (len(params), len(params))

        Raises
        ------
        ValueError
            If full_cov and reinterpret_batch are True
        ValueError
            If querying the emulator outside of its trained grid points
        """
        params = torch.atleast_2d(params)

        if full_cov and reinterpret_batch:
            raise ValueError(
                "Cannot reshape the full_covariance matrix for many parameters."
            )

        if not self._trained:
            warnings.warn(
                "This emulator has not been trained and therefore is not reliable. call \
                    emulator.train() to train."
            )

        # If the pars is outside of the range of emulator values, raise a ModelError
        if torch.any(params < self.min_params) or torch.any(params > self.max_params):
            raise ValueError("Querying emulator outside of original parameter range.", params)

        # Do this according to R&W eqn 2.18, 2.19
        # Recalculate V12, V21, and V22.
        v12 = batch_kernel(self.grid_points, params, self.variances, self.lengthscales)
        v22 = batch_kernel(params, params, self.variances, self.lengthscales)
        v21 = v12.T

        # Recalculate the covariance
        mu = v21 @ torch.linalg.solve(self.v11, self.w_hat)
        cov = v22 - v21 @ torch.linalg.solve(self.v11, v12)
        if not full_cov:
            cov = torch.diag(cov)
        if reinterpret_batch:
            mu = reshape_fortran(mu, (-1, self.ncomps)).squeeze()
            cov = reshape_fortran(cov, (-1, self.ncomps)).squeeze()
            # mu = mu.reshape(-1, self.ncomps, order="F").squeeze()
            # cov = cov.reshape(-1, self.ncomps, order="F").squeeze()
        return mu, cov

    @property
    def bulk_fluxes(self) -> NDArray[float]:
        """
        numpy.ndarray: A vertically concatenated vector of the eigenspectra, flux_mean,
        and flux_std (in that order). Used for bulk processing with the emulator.
        """
        return torch.vstack([self.eigenspectra, self.flux_mean, self.flux_std])

    def load_flux(
        self, params: Union[Sequence[float], NDArray[float]], norm=False
    ) -> NDArray[float]:
        """
        Interpolate a model given any parameters within the grid's parameter range
        using eigenspectrum reconstruction
        by sampling from the weight distributions.

        Parameters
        ----------
        params : array_like
            The parameters to sample at.

        Returns
        -------
        flux : numpy.ndarray
        """
        mu, cov = self(params, reinterpret_batch=False)
        weights = torch.DoubleTensor(np.random.multivariate_normal(mu.numpy(), cov.numpy()).reshape(-1, self.ncomps))
        # weights = torch.distributions.MultivariateNormal(mu, cov).sample().reshape(-1, self.ncomps)
        X = self.eigenspectra * self.flux_std
        flux = weights @ X + self.flux_mean
        if norm:
            flux *= self.norm_factor(params).unsqueeze(1)
        return flux.squeeze()

    def norm_factor(self, params: Union[Sequence[float], NDArray[float]]) -> float:
        """
        Return the scaling factor for the absolute flux units in flux-normalized spectra

        Parameters
        ----------
        params : array_like
            The parameters to interpolate at

        Returns
        -------
        factor: float
            The multiplicative factor to normalize a spectrum to the model's absolute flux units
        """
        _params = np.asarray(params)
        return torch.DoubleTensor(self.factor_interpolator(_params))

    def determine_chunk_log(self, wavelength: Sequence[float], buffer: float = 50):
        """
        Possibly truncate the wavelength and eigenspectra in response to some new
        wavelengths

        Parameters
        ----------
        wavelength : array_like
            The new wavelengths to truncate to
        buffer : float, optional
            The wavelength buffer, in Angstrom. Default is 50

        See Also
        --------
        Starfish.grid_tools.utils.determine_chunk_log
        """
        wavelength = np.asarray(wavelength)

        # determine the indices
        wl_min = wavelength.min()
        wl_max = wavelength.max()

        wl_min -= buffer
        wl_max += buffer

        ind = determine_chunk_log(self.wl, wl_min, wl_max)
        trunc_wavelength = self.wl[ind]

        assert (trunc_wavelength.min() <= wl_min) and (
            trunc_wavelength.max() >= wl_max
        ), (
            f"Emulator chunking ({trunc_wavelength.min():.2f}, {trunc_wavelength.max():.2f}) didn't encapsulate "
            f"full wl range ({wl_min:.2f}, {wl_max:.2f})."
        )

        self.wl = trunc_wavelength
        self.eigenspectra = self.eigenspectra[:, ind]

    def train(self, **opt_kwargs):
        """
        Trains the emulator's hyperparameters using gradient descent. This is a light wrapper around `scipy.optimize.minimize`. If you are experiencing problems optimizing the emulator, consider implementing your own training loop, using this function as a template.

        Parameters
        ----------
        **opt_kwargs
            Any arguments to pass to the optimizer. By default, `method='Nelder-Mead'`
            and `maxiter=10000`.

        See Also
        --------
        scipy.optimize.minimize

        """

        # Do the optimization
        # Enable autograd for parameters
        self.hyperparams.requires_grad_(True)
        # Keep track of parameters
        params = self.hyperparams.params()

        variance_params = self.hyperparams['log_variance'].values()[0]
        lengthscale_params = self.hyperparams['log_lengthscale'].values()[0]
        

        default_kwargs = {"maxiter": 10000, "log_interval": 100, 'best_eps': 1e-5, 'lr': 0.001, 'momentum': 0.9}
        default_kwargs.update(opt_kwargs)

        optimizer = torch.optim.SGD(params, lr = default_kwargs['lr'], momentum = default_kwargs['momentum'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # best_params = {k: v.detach() for k, v in self.hyperparams.items()}
        early_stopping = 20
        best_loss = None
        best_step = 0

        for step in range(default_kwargs['maxiter']):
            optimizer.zero_grad()

            variances = torch.exp(variance_params)
            lengthscales = torch.exp(lengthscale_params).reshape(self.ncomps, -1)
            lambda_xi = torch.exp(self.hyperparams['log_lambda_xi'])
            
            self.v11 = self.iPhiPhi / lambda_xi + batch_kernel(
                self.grid_points, self.grid_points, variances, lengthscales
            )
            loss = -self.log_likelihood()
            
            if best_loss is None or best_loss - loss > default_kwargs['best_eps'] :
                best_loss = loss
                best_step = step
                # best_params = {k: v.detach() for k, v in self.hyperparams.items()}

            if step % default_kwargs["log_interval"] == 0:
                self.log.info(f"step: {step} loss: {loss.item()} (best loss: {best_loss if best_loss is None else best_loss.item()} @ step: {best_step})")
            else:
                self.log.debug(f"step: {step} loss: {loss.item()} (best loss: {best_loss if best_loss is None else best_loss.item()} @ step: {best_step})")

            if step - best_step > early_stopping:
                self.log.info(f"Early stopping as step: {step} loss: {loss} (best loss: {best_loss} @ step: {best_step})")
                break
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        
        self.log.info("Finished optimizing emulator hyperparameters")
        self.hyperparams.requires_grad_(False)
        self._trained = True
        self.log.info(self)
        # self.hyperparams._values = best_params.detach()
        # for k, v in best_params.items():
        #     self.hyperparams[k] = v
        # variances = torch.exp(variance_params).detach()
        # lengthscales = torch.exp(lengthscale_params).reshape(self.ncomps, -1).detach()
        variance_params = self.hyperparams['log_variance'].values()[0]
        lengthscale_params = self.hyperparams['log_lengthscale'].values()[0]
        variances = torch.exp(variance_params)
        lengthscales = torch.exp(lengthscale_params).reshape(self.ncomps, -1)
        # Recalculate v11 and detach
        self.v11 = self.iPhiPhi / self.lambda_xi + batch_kernel(
            self.grid_points, self.grid_points, variances, lengthscales
        ).detach()

    def get_index(self, params: Sequence[float]) -> int:
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, grid_points, and weights.

        Parameters
        ----------
        params : array_like
            The stellar parameters

        Returns
        -------
        index : int

        """
        params = np.atleast_2d(params)
        marks = np.abs(self.grid_points - np.expand_dims(params, 1)).sum(axis=-1)
        return marks.argmin(axis=1).squeeze()

    def get_param_dict(self) -> dict:
        """
        Gets the dictionary of parameters. This is the same as `Emulator.params`

        Returns
        -------
        dict
        """
        return self.hyperparams

    def set_param_dict(self, params: dict):
        """
        Sets the parameters with a dictionary

        Parameters
        ----------
        params : dict
            The new parameters.
        """
        for key, val in params.items():
            if key in self.hyperparams:
                self.hyperparams[key] = val

        self.v11 = self.iPhiPhi / self.lambda_xi + batch_kernel(
            self.grid_points, self.grid_points, self.variances, self.lengthscales
        )

    def log_likelihood(self) -> float:
        """
        Get the log likelihood of the emulator in its current state as calculated in
        the appendix of Czekala et al. (2015)

        Returns
        -------
        float

        Raises
        ------
        scipy.linalg.LinAlgError
            If the Cholesky factorization fails
        """
        L = torch.linalg.cholesky(self.v11)
        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
        sqmah = self.w_hat @ torch.cholesky_solve(self.w_hat.reshape((-1, 1)), L)
        return -(logdet + sqmah) / 2

    def __repr__(self):
        output = "Emulator\n"
        output += "-" * 8 + "\n"
        if self.name is not None:
            output += f"Name: {self.name}\n"
        output += f"Trained: {self._trained}\n"
        output += f"lambda_xi: {self.lambda_xi.item():.2f}\n"
        output += "Variances:\n"
        output += "\n".join([f"\t{v:.2f}" for v in self.variances])
        output += "\nLengthscales:\n"
        output += "\n".join(
            [
                "\t[ " + " ".join([f"{l:.2f} " for l in ls]) + "]"
                for ls in self.lengthscales
            ]
        )
        output += f"\nLog Likelihood: {self.log_likelihood().item():.2f}\n"
        return output
