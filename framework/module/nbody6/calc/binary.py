import astropy.units as u
import numpy as np
from astropy.constants import G, L_sun, M_sun, R_sun, sigma_sb


def calc_semi_major_axis(m1: float, m2: float, period_days: float) -> float:
    """
    Calculate the semi-major axis of a binary system given the masses and orbital period.
    
    :param m1: Mass of the first star in solar masses.
    :param m2: Mass of the second star in solar masses.
    :param period_days: Orbital period in days.
    :return: Semi-major axis in astronomical units (AU).
    """
    total_mass = (m1 + m2) * M_sun
    period = period_days * u.day

    a = ((G * total_mass * period**2) / (4 * np.pi**2)) ** (1 / 3)
    return float(a.to(u.AU).value)


def calc_effective_luminosity(L1: float, L2: float) -> float:
    """
    Calculate the effective luminosity of a binary system as if it were a single star.
    
    :param L1: Luminosity of the first star in solar luminosities.
    :param L2: Luminosity of the second star in solar luminosities.
    :return: Effective luminosity in solar luminosities.
    """
    return L1 + L2


def calc_effective_radius(r1: float, r2: float) -> float:
    """
    Calculate the effective radius of a binary system as if it were a single star.
    The "effective radius" is defined as the radius that would result in the same surface area
    as the combined surface areas of the two stars.
    
    :param r1: Radius of the first star in solar radii.
    :param r2: Radius of the second star in solar radii.
    :return: Effective radius in solar radii.
    """
    return np.sqrt(np.power(r1, 2) + np.power(r2, 2))


def calc_effective_temperature(L: float, R: float) -> float:
    """
    Calculate the effective temperature of a start, given its luminosity and radius.
    Based on the Stefan-Boltzmann law:
    :math:`L = 4 \pi R^2 \sigma T^4`, where `L` is luminosity, `R` is radius, `T` is temperature and `\sigma` is the Stefan-Boltzmann constant.
    The effective temperature is given by:
    :math:`T = \left( \frac{L}{4 \pi R^2 \sigma} \right)^{1/4}`.
    
    :param L: Luminosity in solar luminosities.
    :param R: Radius in solar radii.
    :return: Effective temperature in Kelvin.
    """
    return (L * L_sun / (4 * np.pi * R**2 * R_sun**2 * sigma_sb)) ** 0.25


if __name__ == "__main__":
    m1, m2 = 1, 0.5
    period_days = 365.25

    semi_major_axis = calc_semi_major_axis(m1, m2, period_days)
    print(f"Semi-major axis: {semi_major_axis:.3f} AU")
