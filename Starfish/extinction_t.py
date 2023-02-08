import torch
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Cardelli, Clayton & Mathis (1989)

def ccm89ab_ir_invum(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = x**1.61
    # return a, b
    a = 0.574 * y
    b = -0.527 * y
    return a, b

def ccm89ab_opt_invum(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = x - 1.82
    a = ((((((0.329990*y - 0.77530)*y + 0.01979)*y + 0.72085)*y - 0.02427)*y
             - 0.50447)*y + 0.17699)*y + 1.0
    b = ((((((-2.09002*y + 5.30260)*y - 0.62251)*y - 5.38434)*y + 1.07233)*y
             + 2.28305)*y + 1.41338)*y
    return a, b

def ccm89ab_uv_invum(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ccm89 a, b parameters for 3.3 < x < 8.0 (ultraviolet)"""
    y = x - 4.67
    a = 1.752 - 0.316*x - (0.104 / (y*y + 0.341))
    y = x - 4.62
    b = -3.090 + 1.825*x + (1.206 / (y*y + 0.263))
    mask = x > 5.9
    
    y[mask] = x[mask] - 5.9
    y2 = y[mask] * y[mask]
    y3 = y2 * y[mask]
    a[mask] += -0.04473*y2 - 0.009779*y3
    b[mask] += 0.2130*y2 + 0.1207*y3
    return a, b


def ccm89ab_fuv_invum(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ccm89 a, b parameters for 8 < x < 11 (far-UV)"""

    y = x - 8.
    y2 = y * y
    y3 = y2 * y
    a = -0.070*y3 + 0.137*y2 - 0.628*y - 1.073
    b = 0.374*y3 - 0.420*y2 + 4.257*y + 13.670
    return a, b


def ccm89ab_invum(x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ccm89 a, b parameters for 0.3 < x < 11 in microns^-1"""
    ir_mask = x < 1.1
    opt_mask = ~ir_mask & (x < 3.3)
    uv_mask = ~ir_mask & ~opt_mask & (x < 8)
    fuv_mask = ~(ir_mask | opt_mask | uv_mask)

    a, b = torch.zeros_like(x, device = x.device), torch.zeros_like(x, device = x.device)
    a[ir_mask], b[ir_mask] = ccm89ab_ir_invum(x[ir_mask])
    a[opt_mask], b[opt_mask] = ccm89ab_opt_invum(x[opt_mask])
    a[uv_mask], b[uv_mask] = ccm89ab_uv_invum(x[uv_mask])
    a[fuv_mask], b[fuv_mask] = ccm89ab_fuv_invum(x[fuv_mask])

    return a, b

def ccm89(wave:torch.Tensor, a_v:float, r_v:float, unit:str = 'aa', out:Optional[torch.Tensor]=None) -> torch.Tensor:
    """ccm89(wave, a_v, r_v, unit='aa', out=None)

    Cardelli, Clayton & Mathis (1989) extinction function.

    The parameters given in the original paper [1]_ are used.
    The claimed validity is 1250 Angstroms to 3.3 microns.

    Parameters
    ----------
    wave : torch.Tensor (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).
    out : torch.Tensor, optional
        If specified, store output values in this array.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    Notes
    -----
    In Cardelli, Clayton & Mathis (1989) the mean
    R_V-dependent extinction law, is parameterized as

    .. math::

       <A(\lambda)/A_V> = a(x) + b(x) / R_V

    where the coefficients a(x) and b(x) are functions of
    wavelength. At a wavelength of approximately 5494.5 angstroms (a
    characteristic wavelength for the V band), a(x) = 1 and b(x) = 0,
    so that A(5494.5 angstroms) = A_V. This function returns

    .. math::

       A(\lambda) = A_V (a(x) + b(x) / R_V)

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245

    """

    n = wave.shape[0]

    if out is None:
        out = torch.empty(n, dtype=torch.float64, device = wave.device)
    else:
        assert out.shape == wave.shape
        assert out.dtype == torch.float64

    if unit == 'aa':
        wave = 1e4 / wave
    elif unit != 'invum':
        raise ValueError("unrecognized unit")

    a, b = ccm89ab_invum(wave)

    out[:] = a_v * (a + b / r_v)

    # for i in range(n): # for each wavelength
    #     a, b = ccm89ab_invum(wave[i])
    #     out[i] = a_v * (a + b / r_v)

    return out

# -----------------------------------------------------------------------------
# O'Donnell (1994)

def od94ab_opt_invum(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """od94 a, b parameters for 1.1 < x < 3.3 (optical)"""

    y = x - 1.82
    a = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y +
              0.701)*y - 0.609)*y + 0.104)*y + 1.0
    b = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y -
              3.989)*y + 2.908)*y + 1.952)*y
    return a, b

def od94ab_invum(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ccm89 a, b parameters for 0.3 < x < 11 in microns^-1"""
    ir_mask = x < 1.1
    opt_mask = ~ir_mask & (x < 3.3)
    uv_mask = ~ir_mask & ~opt_mask & (x < 8)
    fuv_mask = ~(ir_mask | opt_mask | uv_mask)

    a, b = torch.zeros_like(x), torch.zeros_like(x)
    a[ir_mask], b[ir_mask] = ccm89ab_ir_invum(x[ir_mask])
    a[opt_mask], b[opt_mask] = od94ab_opt_invum(x[opt_mask])
    a[uv_mask], b[uv_mask] = ccm89ab_uv_invum(x[uv_mask])
    a[fuv_mask], b[fuv_mask] = ccm89ab_fuv_invum(x[fuv_mask])

    return a, b

def odonnell94(wave: torch.Tensor, a_v:float, r_v:float, unit:str='aa', out:Optional[torch.Tensor]=None) -> torch.Tensor:
    """odonnell94(wave, a_v, r_v, unit='aa', out=None)

    O'Donnell (1994) extinction function.

    Like Cardelli, Clayton, & Mathis (1989) [1]_ but using the O'Donnell
    (1994) [2]_ optical coefficients between 3030 A and 9091 A.

    Parameters
    ----------
    wave : torch.DoubleTensor (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).
    out : torch.Tensor, optional
        If specified, store output values in this array.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    Notes
    -----
    This function matches the Goddard IDL astrolib routine CCM_UNRED.
    From the documentation for that routine:

    1. The CCM curve shows good agreement with the Savage & Mathis (1979)
       [3]_ ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.

    2. Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989) [4]_

    3. Valencic et al. (2004) [5]_ revise the ultraviolet CCM
       curve (3.3 -- 8.0 um^-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
    .. [2] O'Donnell, J. E. 1994, ApJ, 422, 158O 
    .. [3] Savage & Mathis 1979, ARA&A, 17, 73
    .. [4] Longo et al. 1989, ApJ, 339,474
    .. [5] Valencic et al. 2004, ApJ, 616, 912

    """


    n = wave.shape[0]

    if out is None:
        out = torch.empty(n, dtype=wave.dtype, device = wave.device)
    else:
        assert out.shape == wave.shape
        assert out.dtype == wave.dtype

    if unit == 'aa':
        wave = 1e4 / wave
    elif unit != 'invum':
        raise ValueError("unrecognized unit")

    a, b = od94ab_invum(wave)

    out[:] = a_v * (a + b / r_v)

    return out

# -----------------------------------------------------------------------------
# Calzetti 2000
# http://adsabs.harvard.edu/abs/2000ApJ...533..682C

def calzetti00k_uv_invum(x:torch.Tensor) -> torch.Tensor:
    """calzetti00 `k` for 0.12 microns < wave < 0.63 microns (UV/optical),
    x in microns^-1"""
    return 2.659 * (((0.011*x - 0.198)*x + 1.509)*x - 2.156)


def calzetti00k_ir_invum(x:torch.Tensor) -> torch.Tensor:
    """calzetti00 `k` for 0.63 microns < wave < 2.2 microns (optical/IR),
    x in microns^-1"""
    return 2.659 * (1.040*x - 1.857)


def calzetti00_invum(x:torch.Tensor, r_v:float) -> torch.Tensor:
    k = None
    mask = x > 1.5873015873015872
    k = torch.zeros_like(x)
    k[mask] = calzetti00k_uv_invum(x[mask])
    k[~mask] = calzetti00k_ir_invum(x[~mask])

    return 1.0 + k / r_v


def calzetti00(wave:torch.Tensor, a_v:float, r_v:float, unit:str='aa', out:Optional[torch.Tensor] = None) -> torch.Tensor:
    """calzetti00(wave, a_v, r_v, unit='aa', out=None)

    Calzetti (2000) extinction function.

    Calzetti et al. (2000, ApJ 533, 682) developed a recipe for
    dereddening the spectra of galaxies where massive stars dominate the
    radiation output, valid between 0.12 to 2.2 microns. They estimate
    :math:`R_V = 4.05 \pm 0.80` from optical-IR observations of
    4 starburst galaxies.

    Parameters
    ----------
    wave : torch.Tensor (1-d)
        Wavelengths or wavenumbers.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band  wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).
    unit : {'aa', 'invum'}, optional
        Unit of wave: 'aa' (Angstroms) or 'invum' (inverse microns).
    out : torch.Tensor, optional
        If specified, store output values in this array.

    Returns
    -------
    Extinction in magnitudes at each input wavelength.

    """

    n = wave.shape[0]

    if out is None:
        out = torch.empty(n, dtype=torch.float64, device = wave.device)
    else:
        assert out.shape == wave.shape
        assert out.dtype == torch.float64

    if unit == 'aa':
        wave = 1e4 / wave
    elif unit != 'invum':
        raise ValueError("unrecognized unit")


    out[:] = a_v * calzetti00_invum(wave, r_v)

    return out