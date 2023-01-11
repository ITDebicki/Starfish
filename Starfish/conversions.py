def microns_to_angstrom(x):
    return x * 10000

def w_m2_microns_to_erg_cm2_s_cm(x):
    return w_m2_to_erg_cm2_s(x) / 10000

def w_m2_to_erg_cm2_s(x):
    return x * 1000