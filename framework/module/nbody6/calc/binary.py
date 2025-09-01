from typing import List, Union

import astropy.units as u
import numpy as np
from astropy.constants import G, M_sun


def calc_semi_major_axis(m1_sol: float, m2_sol: float, period_days: float) -> float:
    """
    Calculate the semi-major axis of a binary system given the masses and orbital period.

    :param m1_sol: Mass of the first star in solar masses.
    :param m2_sol: Mass of the second star in solar masses.
    :param period_days: Orbital period in days.
    :return: Semi-major axis in astronomical units (AU).
    """
    total_mass = (m1_sol + m2_sol) * M_sun
    period = period_days * u.day

    a = ((G * total_mass * period**2) / (4 * np.pi**2)) ** (1 / 3)
    return float(a.to(u.AU).value)


def calc_total_luminosity(L1_sol: float, L2_sol: float) -> float:
    """
    Calculate the effective luminosity of a binary system as if it were a single star.

    :param L1_sol: Luminosity of the first star in solar luminosities.
    :param L2_sol: Luminosity of the second star in solar luminosities.
    :return: Effective luminosity in solar luminosities.
    """
    return L1_sol + L2_sol


def calc_log_total_luminosity(log_L1_sol: float, log_L2_sol: float) -> float:
    """
    Calculate the effective luminosity of a binary system as if it were a single star,
    given the logarithm (base 10) of the luminosities.

    :param log_L1_sol: Logarithm (base 10) of the luminosity of the first star in solar luminosities.
    :param log_L2_sol: Logarithm (base 10) of the luminosity of the second star in solar luminosities.
    :return: Logarithm (base 10) of the effective luminosity in solar luminosities.
    """
    return np.log10(
        calc_total_luminosity(
            L1_sol=np.power(10, log_L1_sol), L2_sol=np.power(10, log_L2_sol)
        )
    )


def calc_total_mass(m1_sol: float, m2_sol: float) -> float:
    """
    Calculate the total mass of a binary system.

    :param m1_sol: Mass of the first star in solar masses.
    :param m2_sol: Mass of the second star in solar masses.
    :return: Total mass in solar masses.
    """
    return m1_sol + m2_sol


def calc_equivalent_radius(r1_sol: float, r2_sol: float) -> float:
    """
    Calculate the effective radius of a binary system as if it were a single star.
    The "effective radius" is defined as the radius that would result in the same surface area
    as the combined surface areas of the two stars.

    :param r1_sol: Radius of the first star in solar radii.
    :param r2_sol: Radius of the second star in solar radii.
    :return: Effective radius in solar radii.
    """
    return np.sqrt(np.power(r1_sol, 2) + np.power(r2_sol, 2))


def calc_photocentric(
    L1_sol: float,
    L2_sol: float,
    vec1: Union[np.ndarray, List[float]],
    vec2: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """
    Calculate the photocentric vector (position or velocity) of a binary system.
    This works for positions, velocities, or any other vector quantity that
    should be luminosity-weighted.

    :param L1_sol: Luminosity of the first star in solar luminosities.
    :param L2_sol: Luminosity of the second star in solar luminosities.
    :param vec1: Vector quantity of the first star (e.g., position [x1, y1, z1] or velocity [vx1, vy1, vz1]).
    :param vec2: Vector quantity of the second star (e.g., position [x2, y2, z2] or velocity [vx2, vy2, vz2]).
    :return: Luminosity-weighted photocentric vector as a numpy array.
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    denom = L1_sol + L2_sol
    if denom == 0:
        raise ValueError("Total luminosity (L1_sol + L2_sol) must be > 0.")
    return (L1_sol * vec1 + L2_sol * vec2) / denom


if __name__ == "__main__":
    m1, m2 = 1, 0.5
    period_days = 365.25

    semi_major_axis = calc_semi_major_axis(m1, m2, period_days)
    print(f"Semi-major axis: {semi_major_axis:.3f} AU")
