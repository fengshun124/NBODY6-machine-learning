import numpy as np
from astropy.constants import L_bol0, L_sun, R_sun, sigma_sb


def calc_effective_temperature(L_sol: float, R_sol: float) -> float:
    """
    Calculate the effective temperature of a star, given its luminosity and radius.
    Based on the Stefan-Boltzmann law:
    :math:`L = 4 \pi R^2 \sigma T^4`, where `L` is luminosity, `R` is radius, `T` is temperature and `\sigma` is the Stefan-Boltzmann constant.
    The effective temperature is given by:
    :math:`T = \left( \frac{L}{4 \pi R^2 \sigma} \right)^{1/4}`.

    :param L_sol: Luminosity in solar luminosities.
    :param R_sol: Radius in solar radii.
    :return: Effective temperature in Kelvin (float).
    """
    R = R_sol * R_sun
    L = L_sol * L_sun
    T = (L / (4 * np.pi * R**2 * sigma_sb)) ** 0.25
    return T.to_value("K")


def calc_log_effective_temperature(log_L_sol: float, log_R_sol: float) -> float:
    """
    Calculate the effective temperature of a star, given its luminosity and radius in logarithmic scale (base 10).
    Based on the Stefan-Boltzmann law:
    :math:`L = 4 \pi R^2 \sigma T^4`, where `L` is luminosity, `R` is radius, `T` is temperature and `\sigma` is the Stefan-Boltzmann constant.
    The effective temperature is given by:
    :math:`T = \left( \frac{L}{4 \pi R^2 \sigma} \right)^{1/4}`.

    :param log_L_sol: Logarithm (base 10) of luminosity in solar luminosities.
    :param log_R_sol: Logarithm (base 10) of radius in solar radii.
    :return: Logarithm (base 10) of effective temperature in Kelvin.
    """
    return np.log10(
        calc_effective_temperature(
            L_sol=np.power(10, log_L_sol), R_sol=np.power(10, log_R_sol)
        )
    )


def calc_log_surface_flux(log_T_eff: float, log_T_sun: float = np.log10(5772)) -> float:
    """
    Calculate the logarithm (base 10) of the surface flux of a star, given its effective temperature.
    The surface flux is given by the Stefan-Boltzmann law:
    :math:`F = \sigma T^4`, where `F` is the surface flux, `T` is the effective temperature and `\sigma` is the Stefan-Boltzmann constant.

    :param log_T_eff: Logarithm (base 10) of effective temperature in Kelvin.
    :param log_T_sun: Logarithm (base 10) of the Sun's effective temperature in Kelvin. Default is `np.log10(5772)`.
    :return: Logarithm (base 10) of surface flux in solar surface flux units.
    """
    return 4 * (np.asarray(log_T_eff) - log_T_sun)


def calc_bolometric_magnitude(log_L_sol: float) -> float:
    """
    Calculate the bolometric magnitude of a star given its luminosity in solar units in logarithmic scale (base 10).
    The bolometric magnitude is defined as
    :math:`M_{bol} = M_{bol,\odot} - 2.5 \log_{10}(L_\star / L_{\odot})`,
    where `M_{bol,\odot}` is the bolometric magnitude of the Sun (-2.5 * log10(L_sun / L_bol0) \approx 4.74),
    `L_\star` is the luminosity of the star, and `L_{\odot}` is the luminosity of the Sun.

    :param log_L_sol: Logarithm (base 10) of luminosity in solar luminosities.
    :return: Bolometric magnitude (float).
    """
    return -2.5 * (log_L_sol + np.log10(L_sun / L_bol0))
