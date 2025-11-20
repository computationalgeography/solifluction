from typing import Any

import lue.framework as lfr


def viscosity_exp_temp(temperature: Any, a: float, b: float) -> Any:
    """
    Calculate viscosity as an exponential function of temperature.

    The viscosity decreases exponentially with increasing temperature according
    to the formula: viscosity = a * exp(-b * temperature).

    Parameters
    ----------
    temperature : Any
        Temperature at which to calculate the viscosity.
    a : float
        Pre-exponential factor (viscosity at zero temperature).
    b : float
        Temperature sensitivity factor.

    Returns
    -------
    Any
        Viscosity corresponding to the given temperature.
    """

    return a * lfr.exp(-temperature * b)


def viscosity_vegetation(
    viscosity_not_vegetated: Any,
    viscosity_fully_vegetated: Any,
    vegetation_fraction: Any,
) -> Any:
    """
    Calculate the effective viscosity based on vegetation cover using linear interpolation.

    Parameters
    ----------
    viscosity_not_vegetated : Any
        Viscosity of the surface without vegetation.
    viscosity_fully_vegetated : Any
        Viscosity of the surface with full vegetation cover.
    vegetation_fraction : Any
        Fraction of vegetation cover (0 for no vegetation, 1 for fully vegetated).

    Returns
    -------
    Any
        Effective viscosity accounting for vegetation fraction.
    """

    return ((1 - vegetation_fraction) * viscosity_not_vegetated) + (
        vegetation_fraction * viscosity_fully_vegetated
    )
