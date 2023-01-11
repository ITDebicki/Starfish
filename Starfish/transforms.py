import extinction  # This may be marked as unused, but is necessary

from Starfish.constants import c_kms
from Starfish.utils import calculate_dv
from . import extinction_t
import torch



def h_poly(t):
    # Just store the inverse matrix (output of:)
    # torch.inverse(torch.tensor(
    #     [[1, 0, 0, 0],
    #     [1, 1, 1, 1],
    #     [0, 1, 0, 0],
    #     [0, 1, 2, 3]]
    # , dtype=float))
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=float, device = t.device)
    # Calculate each value to the power 0, 1, 2 & 3
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    # Multiply by inverse to find result
    return A @ tt

def resample(x, y, xs):
    """
    Resample onto a new wavelength grid using k=5 spline interpolation

    Parameters
    ----------
    x : array_like
        The original wavelength grid
    y : array_like
        The fluxes to resample
    xs : array_like
        The new wavelength grid

    Raises
    ------
    ValueError
        If the new wavelength grid is not strictly increasing monotonic

    Returns
    -------
    numpy.ndarray
        The resampled flux with the same 1st dimension as the input flux
    """
    # # Check for strictly increasing monotonic
    # print(xs)
    # if not torch.all(xs[1:] - xs[:-1] > 0):
    #     raise ValueError("New wavelength grid must be strictly increasing monotonic")
    # Finite difference
    # (p_(k+1) - p_(k)) / (x_(k+1) - x_(k))
    if y.ndim > 1:
        y = y.T # Change y from (batch, y) to (y, batch)
        m = (y[1:] - y[:-1]) / ((x[1:] - x[:-1]).unsqueeze(0).expand((y.shape[1], -1)).T)
    else:
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])

    # One sided difference at edge of dataset
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        
    # idxs contains the mapping to t parameters
    idxs = torch.searchsorted(x[1:], xs)
    # The size of the gap for the parameter
    dx = (x[idxs + 1] - x[idxs])

    # Calculate offset for each new point in terms of t
    hh = h_poly((xs - x[idxs]) / dx)
    # Multiply out values
    # 
    if y.ndim > 1:
        dx = dx.unsqueeze(0).expand((y.shape[1], -1)).T
        hh = hh.unsqueeze(1).expand((-1, y.shape[1], -1)).transpose(1, 2)


    out = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    if y.ndim > 1:
        return out.T
    else:
        return out

def instrumental_broaden(wave, flux, fwhm):
    """
    Broadens given flux by convolving with a Gaussian kernel appropriate for a
    spectrograph's instrumental properties. Follows the given equation

    .. math::
        f = f * \\mathcal{F}^{\\text{inst}}_v

    .. math::
        \\mathcal{F}^{\\text{inst}}_v = \\frac{1}{\\sqrt{2\\pi \\sigma^2}} \\exp \\left[-\\frac12 \\left( \\frac{v}{\\sigma} \\right)^2 \\right]

    This is carried out by multiplication in the Fourier domain rather than using a
    convolution function.

    Parameters
    ----------
    wave : array_like
        The current wavelength grid
    flux : array_like
        The current flux
    fwhm : float
        The full width half-maximum of the instrument in km/s. Note that this is
        quivalent to :math:`2.355\\cdot \\sigma`

    Raises
    ------
    ValueError
        If the full width half maximum is negative.

    Returns
    -------
    numpy.ndarray
        The broadened flux with the same shape as the input flux
    """

    if fwhm < 0:
        raise ValueError("FWHM must be non-negative")
    dv = calculate_dv(wave)
    freq = torch.fft.rfftfreq(flux.shape[-1], d=dv, dtype=torch.float64)
    flux_ff = torch.fft.rfft(flux)

    sigma = fwhm / 2.355
    flux_ff *= torch.exp(-2 * (torch.pi * sigma * freq) ** 2)

    flux_final = torch.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def rotational_broaden(wave, flux, vsini):
    """
    Broadens flux according to a rotational broadening kernel from Gray (2005) [1]_

    Parameters
    ----------
    wave : array_like
        The current wavelength grid
    flux : array_like
        The current flux
    vsini : float
        The rotational velocity in km/s

    Raises
    ------
    ValueError
        if `vsini` is not positive

    Returns
    -------
    numpy.ndarray
        The broadened flux with the same shape as the input flux


    .. [1] Gray, D. (2005). *The observation and Analysis of Stellar Photospheres*.
    Cambridge: Cambridge University Press. doi:10.1017/CB09781316036570
    """

    if vsini <= 0:
        raise ValueError("vsini must be positive")

    dv = calculate_dv(wave)
    freq = torch.fft.rfftfreq(flux.shape[-1], dv, dtype=torch.float64)
    flux_ff = torch.fft.rfft(flux)
    
    # Calculate the stellar broadening kernel (Gray 2008)
    # Ensure this is in float64
    ub = 2.0 * torch.pi * vsini * freq
    # Remove 0th frequency
    sb = torch.ones_like(ub)
    ub = ub[1:]

    a = torch.special.bessel_j1(ub) / ub
    b = 3 * torch.cos(ub) / (2 * ub**2)
    c = 3.0 * torch.sin(ub) / (2 * ub**3)

    sb[1:] = a - b + c
    # print(sb, sb.shape)
    flux_ff *= sb
    flux_final = torch.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def doppler_shift(wave, vz):
    """
    Doppler shift a spectrum according to the formula

    .. math::
        \\lambda \\cdot \\sqrt{\\frac{c + v_z}{c - v_z}}

    Parameters
    ----------
    wave : array_like
        The unshifted wavelengths
    vz : float
        The doppler velocity in km/s

    Returns
    -------
    numpy.ndarray
        Altered wavelengths with the same shape as the input wavelengths
    """

    dv = torch.sqrt((c_kms + vz) / (c_kms - vz))
    return wave * dv


def extinct(wave, flux, Av, Rv=3.1, law="ccm89"):
    """
    Extinct a spectrum following one of many empirical extinction laws. This makes use
    of the `extinction` package. In general, it follows the form

    .. math:: f \\cdot 10^{-0.4 A_V \\cdot A_\\lambda(R_V)}

    Parameters
    ----------
    wave : array_like
        The input wavelengths in Angstrom
    flux : array_like
        The input fluxes
    Av : float
        The absolute attenuation
    Rv : float, optional
        The relative attenuation (the default is 3.1, which is the Milky Way average)
    law : str, optional
        The extinction law to use. One of `{'ccm89', 'odonnell94', 'calzetti00',
        'fitzpatrick99', 'fm07'}` (the default is 'ccm89'). Currently 'fitzpatrick99' and 'fm07' are not supported for autograd

    Raises
    ------
    ValueError
        If `law` does not match one of the availabe laws
    ValueError
        If Rv is not positive

    Returns
    -------
    numpy.ndarray
        The extincted fluxes, with same shape as input fluxes.
    """
    laws = {
        "ccm89": extinction_t.ccm89,
        "odonnell94": extinction_t.odonnell94,
        "calzetti00": extinction_t.calzetti00,
        "fitzpatrick99": extinction.fitzpatrick99,
        "fm07": extinction.fm07
    }

    if law not in laws:
        raise ValueError("Invalid extinction law given")
    if Rv <= 0:
        raise ValueError("Rv must be positive")

    law_fn = laws[law]
    if law == "fm07":
        A_l = torch.from_numpy(law_fn(wave.to(torch.float64).numpy(), Av.numpy()))
    elif law == 'fitzpatrick99':
        A_l = torch.from_numpy(law_fn(wave.to(torch.float64).numpy(), Av.numpy(), Rv.numpy()))
    else:
        A_l = law_fn(wave.to(torch.float64), Av, Rv)
    flux_final = flux * 10 ** (-0.4 * A_l)
    return flux_final


def rescale(flux, scale):
    """
    Rescale the given flux via the following equation

    .. math:: f \\cdot \\Omega

    Parameters
    ----------
    flux : array_like
        The input fluxes
    scale : float or array_like
        The scaling factor. If an array, must have same shape as the batch dimension of
        :attr:`flux`

    Returns
    -------
    numpy.ndarray
        The rescaled fluxes with the same shape as the input fluxes
    """
    if isinstance(scale, (int, float)):
        scale = torch.DoubleTensor([scale], device = flux.device)
    scale = torch.atleast_1d(scale)
    if len(scale) > 1:
        scale = scale.unsqueeze(1)
    return flux * scale


def renorm(wave, flux, reference_flux):
    """
    Renormalize one spectrum to another

    This uses the :meth:`rescale` function with a :attr:`log_scale` of

    .. math::

        \\log \\Omega = \\left. \\int{f^{*}(w) dw} \\middle/ \\int{f(w) dw} \\right.

    where :math:`f^{*}` is the reference flux, :math:`f` is the source flux, and the
    integrals are over a common wavelength grid

    Parameters
    ----------
    wave : array_like
        The wavelength grid for the source flux
    flux : array_like
        The flux for the source
    reference_flux : array_like
        The reference source to renormalize to

    Returns
    -------
    numpy.ndarray
        The renormalized flux
    """
    factor = _get_renorm_factor(wave, flux, reference_flux)
    return rescale(flux, factor)


def _get_renorm_factor(wave, flux, reference_flux):
    ref_int = torch.trapz(reference_flux, wave)
    flux_int = torch.trapz(flux, wave, axis=-1)
    return ref_int / flux_int


def chebyshev_correct(wave, flux, coeffs):
    """
    Multiply the input flux by a Chebyshev series in order to correct for
    calibration-level discrepancies.

    Parameters
    ----------
    wave : array-lioke
        Input wavelengths
    flux : array-like
        Input flux
    coeffs : array-like
        The coefficients for the chebyshev series.

    Returns
    -------
    numpy.ndarray
        The corrected flux

    Raises
    ------
    ValueError
        If only processing a single spectrum and the linear coefficient is not 1.
    """
    # have to scale wave to fit on domain [0, 1]
    coeffs = torch.tensor(coeffs, dtype = torch.float64)
    if coeffs.ndim == 1 and coeffs[0] != 1:
        raise ValueError(
            "For single spectrum the linear Chebyshev coefficient (c[0]) must be 1"
        )

    scale_wave = wave / wave.max()
    p = chebval(scale_wave, coeffs)
    return flux * p

def chebval(x, c):

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1 * x
