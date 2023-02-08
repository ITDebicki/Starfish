import astropy.units as u
import astropy.constants as const

def microns_to_angstrom(x):
    return x * 10000

def w_m2_microns_to_erg_cm2_s_cm(x):
    return u.Quantity(x, "W/m^2/micron").to_value('erg/s/cm^2/cm')

def erg_cm2_s_hz_to_erg_s_cm2_cm(flux, wl, wl_units = 'Angstrom'):
    # "erg/cm^2/s/Hz" -> "erg/s/cm^2/cm"
    # Convert wavelength units to cm
    return flux * (u.Quantity(wl, wl_units).to_value('cm')**2 / const.c.to_value('cm/s'))


def f_v_to_f_lambda(flux, wl, wl_units = 'A'):
    pass