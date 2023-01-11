from dataclasses import dataclass
from typing import Tuple

# TODO convert to dataclass

# Convert R to FWHM in km/s by \Delta v = c/R
@dataclass
class Instrument:
    """
    Object describing an instrument. This will be used by other methods for
    processing raw synthetic spectra.

    Parameters
    ----------
    name: string
        name of the instrument
    FWHM: float
        the FWHM of the instrumental profile in km/s
    wl_range: Tuple (low, high)
        wavelength range of instrument
    oversampling: float, optional
        how many samples fit across the :attr:`FWHM`. Default is 4.0
    """

    name: str
    FWHM: float
    wl_range: Tuple[float]
    oversampling: float = 4.0

    def __str__(self):
        """
        Prints the relevant properties of the instrument.
        """
        return (
            "instrument Name: {}, FWHM: {:.1f}, oversampling: {:.0f}, "
            "wl_range: {}".format(
                self.name, self.FWHM, self.oversampling, self.wl_range
            )
        )


class TRES(Instrument):
    """TRES instrument"""

    def __init__(self, name="TRES", FWHM=6.8, wl_range=(3500, 9500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        # sets the FWHM and wl_range


class Reticon(Instrument):
    """Reticon instrument"""

    def __init__(self, name="Reticon", FWHM=8.5, wl_range=(5145, 5250)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class KPNO(Instrument):
    """KNPO instrument"""

    def __init__(self, name="KPNO", FWHM=14.4, wl_range=(6250, 6650)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class SPEX(Instrument):
    """SPEX instrument at IRTF in Hawaii"""

    def __init__(self, name="SPEX", FWHM=150.0, wl_range=(7500, 54000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class SPEX_SXD(SPEX):
    """SPEX instrument at IRTF in Hawaii short mode (reduced wavelength range)"""

    def __init__(self, name="SPEX_SXD"):
        super().__init__(name=name, wl_range=(7500, 26000))


class SPEX_PRISM(SPEX):
    """SPEX instrument at IRTF in Hawaii low-res PRISM mode (reduced wavelength range)"""

    def __init__(self, name="SPEX_SXD"):
        super().__init__(name=name, FWHM=1500, wl_range=(7500, 26000))


class IGRINS(Instrument):
    """IGRINS Instruments Abstract Class"""

    def __init__(self, wl_range, name="IGRINS"):
        super().__init__(name=name, FWHM=7.5, wl_range=wl_range)
        self.air = False


class IGRINS_H(IGRINS):
    """IGRINS H band instrument"""

    def __init__(self, name="IGRINS_H", wl_range=(14250, 18400)):
        super().__init__(name=name, wl_range=wl_range)


class IGRINS_K(IGRINS):
    """IGRINS K band instrument"""

    def __init__(self, name="IGRINS_K", wl_range=(18500, 25200)):
        super().__init__(name=name, wl_range=wl_range)


class ESPaDOnS(Instrument):
    """ESPaDOnS instrument"""

    def __init__(self, name="ESPaDOnS", FWHM=4.4, wl_range=(3700, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class DCT_DeVeny(Instrument):
    """DCT DeVeny spectrograph instrument."""

    def __init__(self, name="DCT_DeVeny", FWHM=105.2, wl_range=(6000, 10000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class WIYN_Hydra(Instrument):
    """WIYN Hydra spectrograph instrument."""

    def __init__(self, name="WIYN_Hydra", FWHM=300.0, wl_range=(5500, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G140M_F070LP(Instrument):
    """JWST NIRSpec G140M/F070LP """
    def __init__(self, name="NIRSpec_G140M_F070LP", FWHM=300.0, wl_range=(9000, 12700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G140M_F100LP(Instrument):
    """JWST NIRSpec G140M/F100LP """
    def __init__(self, name="NIRSpec_G140M_F100LP", FWHM=300.0, wl_range=(9700, 18900)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G235M_F170LP(Instrument):
    """JWST NIRSpec G235M/F170LP """
    def __init__(self, name="NIRSpec_G235M_F170LP", FWHM=300.0, wl_range=(16600, 31700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G395M_F290LP(Instrument):
    """JWST NIRSpec G395M/F290LP """
    def __init__(self, name="NIRSpec_G395M_F290LP", FWHM=300.0, wl_range=(28700, 52700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G140H_F070LP(Instrument):
    """JWST NIRSpec G140H/F070LP """
    def __init__(self, name="NIRSpec_G140M_F070LP", FWHM=111.0, wl_range=(9500, 12700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G140H_F100LP(Instrument):
    """JWST NIRSpec G140H/F100LP """
    def __init__(self, name="NIRSpec_G140H_F100LP", FWHM=111.0, wl_range=(9700, 18900)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G235H_F170LP(Instrument):
    """JWST NIRSpec G235H/F170LP """
    def __init__(self, name="NIRSpec_G235H_F170LP", FWHM=111.0, wl_range=(16600, 31700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_G395H_F290LP(Instrument):
    """JWST NIRSpec G395H/F290LP """
    def __init__(self, name="NIRSpec_G395H_F290LP", FWHM=111.0, wl_range=(28700, 52700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class NIRSpec_PRISM(Instrument):
    """JWST NIRSpec PRISM/CLEAR """
    def __init__(self, name="NIRSpec_PRISM", FWHM=3000.0, wl_range=(6000, 53000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_1A(Instrument):
    """MIRI MRS 1A"""
    def __init__(self, name = "MIRI_MRS_1A", FWHM=90.4, wl_range=(49000, 57400)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_1B(Instrument):
    """MIRI MRS 1B"""
    def __init__(self, name = "MIRI_MRS_1B", FWHM=94.0, wl_range=(56600, 66300)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_1C(Instrument):
    """MIRI MRS 1C"""
    def __init__(self, name = "MIRI_MRS_1C", FWHM=96.8, wl_range=(65300, 76500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_2A(Instrument):
    """MIRI MRS 2A"""
    def __init__(self, name = "MIRI_MRS_2A", FWHM=100.3, wl_range=(75100, 87700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_2B(Instrument):
    """MIRI MRS 2B"""
    def __init__(self, name = "MIRI_MRS_2B", FWHM=109.1, wl_range=(86700, 101300)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_2C(Instrument):
    """MIRI MRS 2C"""
    def __init__(self, name = "MIRI_MRS_2C", FWHM=104.9, wl_range=(100200, 117000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_3A(Instrument):
    """MIRI MRS 3A"""
    def __init__(self, name = "MIRI_MRS_3A", FWHM=118.6, wl_range=(115500, 134700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_3B(Instrument):
    """MIRI MRS 3B"""
    def __init__(self, name = "MIRI_MRS_3B", FWHM=167.6, wl_range=(133400, 155700)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_3C(Instrument):
    """MIRI MRS 3C"""
    def __init__(self, name = "MIRI_MRS_3C", FWHM=151.5, wl_range=(154100, 179800)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_4A(Instrument):
    """MIRI MRS 4A"""
    def __init__(self, name = "MIRI_MRS_4A", FWHM=205.5, wl_range=(177000, 209500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_4B(Instrument):
    """MIRI MRS 4B"""
    def __init__(self, name = "MIRI_MRS_4B", FWHM=178.6, wl_range=(2106900, 244800)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)

class MIRI_MRS_4B(Instrument):
    """MIRI MRS 4B"""
    def __init__(self, name = "MIRI_MRS_4B", FWHM=184.1, wl_range=(241900, 279000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)