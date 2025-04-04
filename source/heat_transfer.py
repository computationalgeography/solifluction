import lue.framework as lfr

from source.derivatives_discretization import (
    dy_backward,
    dy_upwind,
    second_derivatives_in_y,
)


def phase_heat_coeff(T, T_f, k_f, k_u, c_f, c_u, W, W_u, L, rho_d):

    k_heat = lfr.where(
        T < T_f - DT,
        k_f,
        lfr.where(
            (T >= T_f - DT) & (T <= T_f),
            k_f + (((k_u - k_f) / DT) * (T - (T_f - DT))),
        ),
        k_u,
    )

    c_heat = lfr.where(
        T < T_f - DT,
        c_f,
        lfr.where(
            (T >= T_f - DT) & (T <= T_f),
            c_f + (L * rho_d * ((W - W_u) / DT)),
        ),
        c_u,
    )

    return k_heat, c_heat


def update_thermal_diffusivity_coeff(T, gama_soil):

    # (.)_f : frozen
    # (.)_u : unfrozen
    # (.)_s : soil mineral
    # (.)_w : water
    # (.)_i : ice
    T_f = 0
    DT = 5
    W = 2  # fill out this ?
    W_u = 1  # fill out this ?
    L = 1  # fill out this ?
    rho_d = 1  # fill out this ?

    k_f_s = 1  # fill out this ?
    k_u_s = 2  # fill out this ?
    c_f_s = 1  # fill out this ?
    c_u_s = 2  # fill out this ?

    k_f_w = 1  # fill out this ?
    k_u_w = 2  # fill out this ?
    c_f_w = 1  # fill out this ?
    c_u_w = 2  # fill out this ?

    k_f_i = 1  # fill out this ?
    k_u_i = 2  # fill out this ?
    c_f_i = 1  # fill out this ?
    c_u_i = 2  # fill out this ?

    k_heat_s, c_heat_s = phase_heat_coeff(
        T, T_f, k_f_s, k_u_s, c_f_s, c_u_s, W, W_u, L, rho_d
    )
    k_heat_w, c_heat_w = phase_heat_coeff(
        T, T_f, k_f_w, k_u_w, c_f_w, c_u_w, W, W_u, L, rho_d
    )
    k_heat_i, c_heat_i = phase_heat_coeff(
        T, T_f, k_f_i, k_u_i, c_f_i, c_u_i, W, W_u, L, rho_d
    )

    vol_frac_soil = 0.5
    vol_frac_ice = 0.2
    vol_frac_water = 0.3

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

    thermal_diffusivity = k_heat / ((gama_soil / 9.81) * c_heat)

    return thermal_diffusivity


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

    # T_neighbor_layer can be T_up_layer or T_down_layer depend of using forward or backward discretization

    # thermal_diffusivity (k/(rho*c))
    # k: thermal conductivity

    T_center_layer = lfr.where(
        compute_flag,
        T_center_layer
        + (
            (dt / rho_c_center)
            * dy_upwind(T_center_layer, T_up_layer, T_down_layer, dy_up, dy_down)
            * dy_upwind(
                k_center,
                k_up,
                k_down,
                dy_up,
                dy_down,
            )
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
