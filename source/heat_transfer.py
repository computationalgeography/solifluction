import lue.framework as lfr

from source.derivatives_discretization import (
    dy_backward,
    dy_upwind,
    second_derivatives_in_y,
)


def phase_heat_coeff(T, D_T, k_f, k_u, c_f, c_u, W, W_u, L, rho_b):
    """
    Compute the heat conductivity (k) and heat capacity (c)
    as a function of temperature, considering phase change.

    Parameters:
    ----------
    T     : Temperature (in Celsius or Kelvin, depending on input usage).
    T_f   : Freezing temperature (typically 0°C or 273.15K).
    D_T   : Width of the phase change envelope (in Kelvin).
    L     : Latent heat of freezing for water (in J/kg).
    W     : Fractional total water content of the soil by mass.
    W_u   : Unfrozen water content that remains at (T_f - D_T),
            typically taken as 5% of W.
    rho_b : ? maybe Bulk density of the soil (m/V_total)

    '{.}_f' : Refers to frozen-phase properties.
              (e.g., k_f for frozen thermal conductivity, c_f for frozen heat capacity).
    '{.}_u' : Refers to unfrozen-phase properties.
              (e.g., k_u for unfrozen thermal conductivity, c_u for unfrozen heat capacity).

    Returns:
    -------
    k : Effective thermal conductivity.
    c : Effective heat capacity.
    """

    T_f = 0

    k_heat = lfr.where(
        T < T_f - D_T,
        k_f,
        lfr.where(
            (T >= T_f - D_T) & (T <= T_f),
            k_f + (((k_u - k_f) / D_T) * (T - (T_f - D_T))),
        ),
        k_u,
    )

    c_heat = lfr.where(
        T < T_f - D_T,
        c_f,
        lfr.where(
            (T >= T_f - D_T) & (T <= T_f),
            c_f + (L * rho_b * ((W - W_u) / D_T)),
        ),
        c_u,
    )

    return k_heat, c_heat


def update_thermal_diffusivity_coeff(
    T,
    rho_soil,
    D_T,
    W,
    W_u,
    L,
    rho_b,
    k_f_s,
    k_u_s,
    c_f_s,
    c_u_s,
    k_f_w,
    k_u_w,
    c_f_w,
    c_u_w,
    k_f_i,
    k_u_i,
    c_f_i,
    c_u_i,
    vol_frac_soil,
    vol_frac_ice,
    vol_frac_water,
):
    """
    Calculate thermal diffusivity coefficient for volume of materials considering the volume fraction
    of different phases (water, ice, soil (mineral or rock)).

    Parameters:
    ----------
    T           : Temperature.
    rho_soil   : Volume fraction of the soil mineral (solid phase).
    k:
    c:

    Phases:
    -------
    - `{.}_f` : Frozen phase (e.g., ice or soil with ice).
    - `{.}_u` : Unfrozen phase (e.g., liquid water or soil without ice).
    - `{.}_s` : Soil mineral (solid phase).
    - `{.}_w` : Water phase.
    - `{.}_i` : Ice phase.

    Returns:
    -------
    k (heat conductivity) and rho_c (rho*c density multiply heat capacity).

    """

    # D_T = 1
    # W = 2  # fill out this ?
    # W_u = 1  # fill out this ?
    # L = 1  # fill out this ?
    # rho_b = 1  # fill out this ?

    # k_f_s = 1  # fill out this ?
    # k_u_s = 2  # fill out this ?
    # c_f_s = 1  # fill out this ?
    # c_u_s = 2  # fill out this ?

    # k_f_w = 1  # fill out this ?
    # k_u_w = 2  # fill out this ?
    # c_f_w = 1  # fill out this ?
    # c_u_w = 2  # fill out this ?

    # k_f_i = 1  # fill out this ?
    # k_u_i = 2  # fill out this ?
    # c_f_i = 1  # fill out this ?
    # c_u_i = 2  # fill out this ?

    # vol_frac_soil = 0.5  ?
    # vol_frac_ice = 0.2  ?
    # vol_frac_water = 0.3  ?

    k_heat_s, c_heat_s = phase_heat_coeff(
        T, D_T, k_f_s, k_u_s, c_f_s, c_u_s, W, W_u, L, rho_b
    )
    k_heat_w, c_heat_w = phase_heat_coeff(
        T, D_T, k_f_w, k_u_w, c_f_w, c_u_w, W, W_u, L, rho_b
    )
    k_heat_i, c_heat_i = phase_heat_coeff(
        T, D_T, k_f_i, k_u_i, c_f_i, c_u_i, W, W_u, L, rho_b
    )

    k_heat = (
        lfr.pow(k_heat_s, vol_frac_soil)
        * lfr.pow(k_heat_i, vol_frac_ice)
        * lfr.pow(k_heat_w, vol_frac_water)
    )

    c_heat = (
        (c_heat_s * vol_frac_soil)
        + (c_heat_i * vol_frac_ice)
        + (c_heat_w * vol_frac_water)
    )

    rho_c = rho_soil * c_heat

    return k_heat, rho_c


# solve Eq: [d_phi/dt + (adv_coeff * d_phi/dy)  - (diff_coeff * d2_phi/dy2) = rhs]
# Heat transfer 1D equation in y: [d_T/dt - ((1/rho_c) *(d_k/dy * d_T/dy)) - ((k/rho_c) * d2_T/dy2) = 0]


def compute_temperature_1D_in_y(
    k_center,
    k_up,
    k_down,
    rho_c_center,
    T_center_layer,
    T_up_layer,
    T_down_layer,
    dt: float,
    dy_up,
    dy_down,
    compute_flag,  # compute_flag: 1 = compute, 0 = use precomputed_value
    precomputed_value,
):

    # k: thermal conductivity

    T_center_layer = lfr.where(
        compute_flag,
        T_center_layer
        + (
            (dt / rho_c_center)
            * dy_upwind(
                T_center_layer, T_up_layer, T_down_layer, dy_up, dy_down, rho_c_center
            )
            * dy_upwind(k_center, k_up, k_down, dy_up, dy_down, rho_c_center)
        )
        + (
            (dt * k_center / rho_c_center)
            * second_derivatives_in_y(
                T_center_layer,
                T_up_layer,
                T_down_layer,
                dy_up,
                dy_down,
            )
        ),
        precomputed_value,
    )

    return T_center_layer
