#!/usr/bin/env python
import os
import os.path
import sys

import docopt
import lue.framework as lfr
import numpy as np

from input_output import write
from source.boundary_condition import boundary_set

# from VOF import mass_conservation_2D


"""
def save_generation(array, pathname, iteration):
    folder_path = os.path.abspath("results")  # Define the absolute folder path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist
    # Save the generation in the results folder
    lfr.to_gdal(
        array, os.path.join(folder_path, "{}-{}.tif".format(pathname, iteration))
    )
"""

# rhs = (
#         g_sin
#         - ((mu_soil / (gama_soil / 9.81) * du2_dy2))
#         - ((gama_prime / (gama_soil / 9.81)) * dh_dx)
#     )

# dh_dx = (
#         h_total - lfr.focal_sum(h_total, kernel_im1_j)
#     ) / dx  # Note: Upwind method, it is assumed the motion is always from uphill to downhill

#     def momentum_ux(
#     phi: lfr.PartitionedArray,
#     phase_state: lfr.PartitionedArray,
#     dx: float,
#     dz: float,
#     dt: float,
#     lue_u_x: lfr.PartitionedArray,
#     lue_u_z: lfr.PartitionedArray,
#     nu_x: float,
#     nu_z: float,
#     rhs: lfr.PartitionedArray,
#     h_mesh: lfr.PartitionedArray,
#     boundary_loc: lfr.PartitionedArray,
#     boundary_value: lfr.PartitionedArray,
#     results_pathname: str,
# ) -> lfr.PartitionedArray:


def momentum_ux(
    phi,
    phase_state,
    dx: float,
    dz: float,
    dt: float,
    lue_u_x,
    lue_u_z,
    nu_x: float,
    nu_z: float,
    rhs,
    h_mesh,
    boundary_loc,
    boundary_type,
    Dirichlet_boundary_value,
    Neumann_boundary_value,
):

    # lue_u_x : phi
    # lue_u_y : 0
    # nu_x : +(mu_mesh/gama_soil_mesh)
    # rhs_g_sin : g*sin(alfa)
    # gama_prime_mesh: (gama_soil/cos(alfa))-(gama_water*cos(alfa))
    # rhs : g*sin(alfa) - ((gama_soil/cos(alfa))-(gama_water*cos(alfa)))*dh/dx

    # kernel_i_j = np.array(
    #     [
    #         [0, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 0],
    #     ],
    #     dtype=np.uint8,
    # )

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    """
    # kernel_i_jm1   i, j-1
    kernel_i_jm1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    """

    # kernel_i_jm1   i, j-1
    kernel_i_jm1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_ip1_j   i+1, j
    kernel_ip1_j = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    """
    # kernel_i_jp1   i, j+1
    kernel_i_jp1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    """

    # kernel_i_jp1   i, j+1
    kernel_i_jp1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # coeff_map_i_j = (
    #     1
    #     + (-(dt / dx) * lfr.abs(lue_u_x))
    #     + (-(dt / dy) * lfr.abs(lue_u_y))
    #     + (2 * (dt / (dx**2)) * nu_x)
    #     + (2 * (dt / (dy**2)) * nu_y)
    # )

    coeff_map_i_j = (
        1  # check this
        + (-(dt / dx) * lfr.abs(lue_u_x))
        + (-(dt / dz) * lfr.abs(lue_u_z))
        + (2 * (dt / (dx**2)) * nu_x)
        + (2 * (dt / (dz**2)) * nu_z)
    )

    coeff_map_im1_j = lfr.where(
        lue_u_x >= 0,
        ((dt / dx) * lfr.abs(lue_u_x)) + (-(dt / (dx**2)) * nu_x),
        (-(dt / (dx**2)) * nu_x),
    )

    coeff_map_ip1_j = lfr.where(
        lue_u_x < 0,
        ((dt / dx) * lfr.abs(lue_u_x)) + (-(dt / (dx**2)) * nu_x),
        (-(dt / (dx**2)) * nu_x),
    )

    coeff_map_i_jm1 = lfr.where(
        lue_u_z >= 0,
        ((dt / dz) * lfr.abs(lue_u_z)) + (-(dt / (dz**2)) * nu_z),
        (-(dt / (dz**2)) * nu_z),
    )

    coeff_map_i_jp1 = lfr.where(
        lue_u_z < 0,
        ((dt / dz) * lfr.abs(lue_u_z)) + (-(dt / (dz**2)) * nu_z),
        (-(dt / (dz**2)) * nu_z),
    )

    # NOTE: coeff_<.> should be implemented on boundaries, for instance on boundary_tpe 1 (left boundary) phi_i-1,j is located outside of domain and we need forward discretization
    # For now this implementation is ignored as we impose certain boundary conditions on the boundaries which overwrite phi on the boundaries but in the future this should be considered
    #  and for each boundary use exclusive discretization

    phi_all_internal_domain_i_j = (
        (coeff_map_i_j * phi)  # (coeff_map_i_j * lfr.focal_sum(solution, kernel_i_j))
        + (coeff_map_im1_j * lfr.focal_sum(phi, kernel_im1_j))
        + (coeff_map_i_jm1 * lfr.focal_sum(phi, kernel_i_jm1))
        + (coeff_map_ip1_j * lfr.focal_sum(phi, kernel_ip1_j))
        + (coeff_map_i_jp1 * lfr.focal_sum(phi, kernel_i_jp1))
        + (dt * rhs)
    )

    # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen), now vegetation is ignored in phase_state but it is considered in vegetation_vol_fraction

    phi_internal = lfr.where(
        (phase_state != 0) & (h_mesh > 0),  # fluid or unfrozen
        phi_all_internal_domain_i_j,
        0,
    )

    phi = boundary_set(
        phi_internal,
        boundary_loc,
        boundary_type,
        Dirichlet_boundary_value,
        Neumann_boundary_value,
        dx,
        dz,
    )

    return phi, phi_internal


# Eq: dT/dt = thermal_diffusivity_coeff *(d2T/d2x)
def update_in_time_advection_diffusion_T(Layer_list, num_layers, dt, T_surf, T_bed):

    # dy_layers is averaged BUT certainly it is WRONG formula for discretization

    Layer_list[0].T = T_surf
    Layer_list[-1].T = T_bed

    for i in range(1, num_layers):

        dy_layers = (
            Layer_list[i].h_mesh + Layer_list[i - 1].h_mesh + Layer_list[i + 1].h_mesh
        ) / 3

        Layer_list[i].T = Layer_list[i].T + (
            ((dt / (dy_layers**2)) * Layer_list[i].thermal_diffusivity_coeff)
            * ((Layer_list[i + 1].T) - (2 * Layer_list[i].T) + (Layer_list[i - 1].T))
        )


# # Eq: dT/dt = thermal_diffusivity_coeff *(d2T/d2x)
# def update_in_time_advection_diffusion_T(
#     T_center,
#     T_up,
#     T_down,
#     thermal_diffusivity_coeff,
#     dt,
#     dy_layers,
# ):

#     # dy_layers is averaged BUT certainly it is WRONG formula for discretization

#     T_center = T_center + (
#         ((dt / (dy_layers**2)) * thermal_diffusivity_coeff)
#         * ((T_up) - (2 * T_center) + (T_down))
#     )

#     return T_center


def mass_conservation(variable, u_mesh, dt, dx):

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    flux_upstream = lfr.focal_sum((u_mesh * variable), kernel_im1_j)

    variable = variable - ((dt / dx) * (flux_upstream - (u_mesh * variable)))

    return variable


def mass_conservation_2D(
    var, u_x_mesh, u_z_mesh, dt, dx, dz, boundary_loc, boundary_value
):

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # kernel_i_jm1   i, j-1    # check this ???
    kernel_i_jm1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_ip1_j   i+1, j
    kernel_ip1_j = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_i_jp1   i, j+1      # check this ???
    kernel_i_jp1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # Upwind first order method
    # It is assumed that var>=0

    flux_x_upstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
    )
    flux_z_upstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
    )

    # var_internal = (
    #     var
    #     + ((dt / dx) * (flux_x_upstream - (u_x_mesh * var)))
    #     + ((dt / dz) * (flux_z_upstream - (u_z_mesh * var)))
    # )

    var_internal = (
        var
        + ((dt / dx) * (lfr.abs(flux_x_upstream) - (lfr.abs(u_x_mesh) * var)))
        + ((dt / dz) * (lfr.abs(flux_z_upstream) - (lfr.abs(u_z_mesh) * var)))
    )

    """ # Averaged flux or central method
    flux_x_upstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
    )
    flux_z_upstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
    )

    flux_x_downstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
    )
    flux_z_downstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
    )

    var_internal = (
        var
        + ((dt / (2 * dx)) * (flux_x_upstream - flux_x_downstream))
        + ((dt / (2 * dz)) * (flux_z_upstream - flux_z_downstream))
    )
    """

    var = lfr.where(
        boundary_loc,
        boundary_value,
        var_internal,
    )

    net_flux = flux_x_upstream - (u_x_mesh * var)

    return var, flux_x_upstream, net_flux


def update_thermal_diffusivity_coeff(T, gama_soil):

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


def update_mu_soil(vegetation_vol_fraction):

    mu_pure_soil = 10**-3
    mu_pure_vegetation = 10

    mu_soil = ((1 - vegetation_vol_fraction) * mu_pure_soil) + (
        vegetation_vol_fraction * mu_pure_vegetation
    )

    return mu_soil


""" 
def mass_conservation0(h_mesh, u_mesh, dt, dx):

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    uh_upstream = lfr.focal_sum((u_mesh * h_mesh), kernel_im1_j)

    h_mesh = h_mesh - ((dt / dx) * (uh_upstream - (u_mesh * h_mesh)))

    return h_mesh

mass_conservation0 """

""" 
def second_derivatives_in_y_uniform_dy(
    layer_variable_center, layer_variable_up, layer_variable_down, dy_layers
):

    # layer_variable_center:center variable with FDM coeff -2 , layer_variable_up and layer_variable_down around variables with FDM coeff 1
    # NOTE: for surface layer layer_variable_center is one layer below, and layer_variable_up and layer_variable_down are surface layer and two layers below

    d2var_dy2 = (1 / (dy_layers**2)) * (
        (layer_variable_up) - (2 * layer_variable_center) + (layer_variable_down)
    )

    return d2var_dy2

uniform dy_layer """


"""# uniform layer distance in y

def second_derivatives_in_y(
    layer_variable_center, layer_variable_up, layer_variable_down, dy_layers
):

    # layer_variable_center:center variable with FDM coeff -2 , layer_variable_up and layer_variable_down around variables with FDM coeff 1
    # NOTE: for surface layer layer_variable_center is one layer below, and layer_variable_up and layer_variable_down are surface layer and two layers below

    # dy_layers is averaged BUT certainly it is WRONG formula for discretization

    d2var_dy2 = (1 / (dy_layers**2)) * (
        (layer_variable_up) - (2 * layer_variable_center) + (layer_variable_down)
    )

    return d2var_dy2
 """


# Non-uniform layer distance in y
def second_derivatives_in_y(
    layer_variable_center,
    layer_variable_up,
    layer_variable_down,
    dy_layers_up,
    dy_layers_down,
):

    # layer_variable_center:center variable with FDM coeff -2 , layer_variable_up and layer_variable_down around variables with FDM coeff 1
    # NOTE: for surface layer (backward) layer_variable_center is one layer below, and layer_variable_up and layer_variable_down are surface layer and two layers below

    # dy_layers is averaged BUT certainly it is WRONG formula for discretization

    d2var_dy2 = (2 / (dy_layers_up + dy_layers_down)) * (
        ((layer_variable_up - layer_variable_center) / dy_layers_up)
        - (((layer_variable_center - layer_variable_down) / dy_layers_down))
    )

    return d2var_dy2


""" VOF 

def h_mesh_assign_vof(h_total, num_layers, Layer_list, partition_shape):

    h_total_max = lfr.maximum(h_total, partition_shape)

    dy_layers = h_total_max / num_layers

    num_fill_h_mesh = lfr.round(
        h_total / dy_layers
    )  # (number_of_layers -1) in which h_mesh >0. This num exclude top mesh

    top_layer_h_mesh = h_total - (dy_layers * num_fill_h_mesh)

    for i in range(1, num_layers):

        Layer_list[-i].h_mesh = lfr.where(num_fill_h_mesh > 0, dy_layers, 0)
        Layer_list[-i].h_mesh = lfr.where(num_fill_h_mesh == 0, top_layer_h_mesh, 0)
        num_fill_h_mesh = num_fill_h_mesh - 1

    return Layer_list

VOF """


class Layer:
    def __init__(
        self,
        u_x,
        u_z,
        T,
        h_mesh,
        mu_soil,
        density_soil,
        phase_state,
        thermal_diffusivity_coeff,
        vegetation_vol_fraction,
    ):

        self.u_x = u_x  # Velocity in x direction
        self.u_z = u_z  # Velocity in z direction  (y is almost in the gravity direction normal to the bed rock, x and z are layer coordinate parallel to bed rock)
        self.T = T  # Temperature
        self.h_mesh = h_mesh  # soil layer thickness in mesh
        self.mu_soil = mu_soil  # soil viscosity
        self.gama_soil = density_soil  #  soil density
        self.phase_state = phase_state  # Phase state (fluid, solid (ice), vegetation)
        self.thermal_diffusivity_coeff = thermal_diffusivity_coeff
        self.vegetation_vol_fraction = vegetation_vol_fraction


@lfr.runtime_scope
def solifluction_simulate(
    dx,
    dz,
    dt,
    T_surf_file,
    T_initial_file,
    h_total_file,
    mu_soil_initial_file,
    mu_soil_surf_file,
    U_x_surf_file,
    U_x_initial_file,
    gama_soil_surf_file,
    gama_soil_initial_file,
    phase_state_surf_file,
    phase_state_initial_file,
    thermal_diffusivity_coeff_surf_file,
    vegetation_vol_fraction_surf_file,
    vegetation_vol_fraction_initial_file,
    nu_x,
    nu_z,
    nr_time_steps,
    partition_shape,
    results_pathname,
):

    # const values
    domain_slope = np.pi / 10
    g_sin = 9.81 * np.sin(domain_slope)
    gama_saturate = 2680 * 9.81
    gama_water = 9810
    gama_prime = gama_saturate - gama_water

    nu_x = 0  # diffusion coefficient in x for momentum equation (fluid motion)
    nu_z = 0  # diffusion coefficient in z for momentum equation (fluid motion)

    T_bed = 0

    # End: const values

    print("---read_from_gdal_to_lue ------------- ")

    T_surf = lfr.from_gdal(T_surf_file, partition_shape)
    T_initial = lfr.from_gdal(T_initial_file, partition_shape)

    h_total = lfr.from_gdal(h_total_file, partition_shape)

    mu_soil_initial = lfr.from_gdal(
        mu_soil_initial_file, partition_shape
    )  # mu_soil_initial it is used for initialize the internal layers

    mu_soil_surf = lfr.from_gdal(mu_soil_surf_file, partition_shape)

    U_surf = lfr.from_gdal(U_x_surf_file, partition_shape)  # It is set to be zero
    U_initial = lfr.from_gdal(
        U_x_initial_file, partition_shape
    )  # Initial velocity for internal layers. It is set to be zero

    gama_soil_surf = lfr.from_gdal(gama_soil_surf_file, partition_shape)
    gama_soil_initial = lfr.from_gdal(gama_soil_initial_file, partition_shape)

    phase_state_surf = lfr.from_gdal(phase_state_surf_file, partition_shape)

    phase_state_initial = lfr.from_gdal(
        phase_state_initial_file, partition_shape
    )  # It is used initially for phase states of internal layers. Initially all internal phases are assumed to be frozen

    thermal_diffusivity_coeff_surf = lfr.from_gdal(
        thermal_diffusivity_coeff_surf_file, partition_shape
    )

    thermal_diffusivity_coeff_initial = lfr.from_gdal(
        thermal_diffusivity_coeff_surf_file, partition_shape
    )

    vegetation_vol_fraction_surf = lfr.from_gdal(
        vegetation_vol_fraction_surf_file, partition_shape
    )

    vegetation_vol_fraction_initial = lfr.from_gdal(
        vegetation_vol_fraction_initial_file, partition_shape
    )

    u_z = lfr.array_like(h_total, 0.0)
    # u_z = lfr.create_array()

    print("---End read_from_gdal_to_lue ------------- ")

    # Initialize rater

    # mu_soil = lfr.create_partitioned_array(array_shape, partition_shape, value1)

    num_layers: int = 5

    h_mesh_uniform = h_total / num_layers

    # instantiate Layer objects for all layers

    Layer_list = []

    # NOTE: number of layers is 0 to "num_layers" for bed layer to surface layer

    # Assign bed layer properties

    Layer_list.append(
        Layer(
            U_initial,
            T_initial,
            h_mesh_uniform,
            mu_soil_initial,
            density_soil_initial,
            phase_state_initial,
            thermal_diffusivity_coeff_initial,
            vegetation_vol_fraction_initial,
        )
    )

    # Assign internal layers properties

    for i in range(1, num_layers):
        Layer_list.append(
            Layer(
                U_initial,
                T_initial,
                h_mesh_uniform,
                mu_soil_initial,
                density_soil_initial,
                phase_state_initial,
                thermal_diffusivity_coeff_initial,
                vegetation_vol_fraction_initial,
            )
        )

    # Assign surface layer properties

    Layer_list.append(
        Layer(
            U_surf,
            T_surf,
            h_mesh_uniform,
            mu_soil_surf,
            density_soil_surf,
            phase_state_surf,
            thermal_diffusivity_coeff_surf,
            vegetation_vol_fraction_surf,
        )
    )

    # End: instantiate Layer objects for all layers

    # write(c_equ_lue, results_pathname, "c_equ", 0)

    # --------------- End: convert numpy initial variables to lue -----------------------------

    h_total_update = h_total

    time = 0

    for time_step in range(1, nr_time_steps + 1):

        time = time + dt

        h_total = h_total_update

        h_total_update = lfr.array_like(h_total, 0.0)
        # h_total_update = lfr.create_array()

        for i in range(0, num_layers):

            # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen), now vegetation is ignored in phase_state but it is considered in vegetation_vol_fraction
            phase_state = lfr.where((Layer_list[i].T <= 0), 0, phase_state_initial)

            # calculate du2_dy2 for the right hand side of momentum (velocity) equation

            if i == 0:
                du2_dy2 = second_derivatives_in_y(
                    Layer_list[1].U,
                    Layer_list[0].U,
                    Layer_list[2].U,
                    (Layer_list[0].h_mesh + Layer_list[0].h_mesh + Layer_list[0].h_mesh)
                    / 3,
                )

            elif i == num_layers - 1:

                du2_dy2 = second_derivatives_in_y(
                    Layer_list[-2],
                    Layer_list[-3],
                    Layer_list[-1],
                    (
                        Layer_list[-2].h_mesh
                        + Layer_list[-3].h_mesh
                        + Layer_list[-1].h_mesh
                    )
                    / 3,
                )

            else:

                du2_dy2 = second_derivatives_in_y(
                    Layer_list[i].U,
                    Layer_list[i + 1].U,
                    Layer_list[i - 1].U,
                    (
                        Layer_list[i].h_mesh
                        + Layer_list[i - 1].h_mesh
                        + Layer_list[i + 1].h_mesh
                    )
                    / 3,
                )

            # momentum for velocity calculation

            Layer_list[i].U = momentum_ux(
                Layer_list[i].U,
                phase_state,
                dx,
                dz,
                dt,
                Layer_list[i].u_x,
                u_z,
                nu_x,
                nu_z,
                g_sin,
                Layer_list[i].mu_soil,
                Layer_list[i].gama_soil,
                du2_dy2,
                gama_prime,
                h_total,
                Layer_list[i].h_mesh,
                results_pathname,
            )

            # update layers mesh height (h_mesh)
            mass_conservation_2D(
                Layer_list[i].h_mesh,
                Layer_list[i].u_x,
                Layer_list[i].u_z,
                dt,
                dx,
                dz,
                boundary_loc_h,
                boundary_value_h,
            )

            # update vegetation fraction (vegetation_vol_fraction)
            mass_conservation_2D(
                Layer_list[i].vegetation_vol_fraction,
                Layer_list[i].u_x,
                Layer_list[i].u_z,
                dt,
                dx,
                dz,
                boundary_loc_veg,
                boundary_value_veg,
            )

            # update mu (mu_soil)
            Layer_list[i].mu_soil = update_mu_soil(
                Layer_list[i].vegetation_vol_fraction
            )

            # Heat transfer

            Layer_list[0].T = T_surf
            Layer_list[-1].T = T_bed

            if (i != 1) and (i != num_layers - 1):

                Layer_list[i].T = update_in_time_advection_diffusion_T(
                    Layer_list[i].T,
                    Layer_list[i + 1].T,
                    Layer_list[i - 1].T,
                    Layer_list[i].thermal_diffusivity_coeff,
                    dt,
                    (
                        Layer_list[i].h_mesh
                        + Layer_list[i - 1].h_mesh
                        + Layer_list[i + 1].h_mesh
                    )
                    / 3,
                )

            h_total_update = h_total_update + Layer_list[i].h_mesh

            print(" time: ", time, " time_step: ", time_step)

            # write(c, results_pathname, "c_time_step", time_step)
            # write(z, results_pathname, "z_time_step", time_step)

            # time_plot = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            time_plot = [10, 20, 50, 80, 100, 200, 300, 400, 500, 800, 900, 1000]

            if any(abs(time - time_plot_i) < dt for time_plot_i in time_plot):
                write(Layer_list[2].U, results_pathname, "Layer_list[2]_U_time", time)
                write(Layer_list[2].T, results_pathname, "Layer_list[2]_T_time", time)

                pass


usage = """\
Calculate the generations of alive cells according to the Game of Life cellular automaton

Usage:
    {command} <dx> <dy> <dt> <T_surf_file>
        <T_initial_file> <h_total_file> <mu_soil_initial_file> <mu_soil_surf_file> 
        <U_x_surf_file> <U_x_initial_file> <gama_soil_surf_file>
        <gama_soil_initial_file> <phase_state_surf_file> <phase_state_initial_file>
        <thermal_diffusivity_coeff_surf_file> <vegetation_vol_fraction_surf_file> <vegetation_vol_fraction_initial_file>
        <nu_x> <nu_y> <partition_extent> <time_end> <results_pathname>

Options:
    <dx>         mesh size in x direction
    <dy>         mesh size in x direction
    <dt>         time step size
    <T_surf_file>
    <T_initial_file>
    <h_total_file>
    <mu_soil_initial_file>
    <mu_soil_surf_file> 
    <U_x_surf_file>
    <U_x_initial_file>
    <gama_soil_surf_file>
    <gama_soil_initial_file> 
    <phase_state_surf_file> 
    <phase_state_initial_file>
    <thermal_diffusivity_coeff_surf_file>
    <vegetation_vol_fraction_surf_file>
    <vegetation_vol_fraction_initial_file>
    <nu_x>       nu_x x direction diffusion coefficient
    <nu_y>       nu_y y direction diffusion coefficient
    <partition_extent> Size of one side of the partitions
    <time_end> End time for simulation 
    <results_pathname> complete pathname to write results 
""".format(
    command=os.path.basename(sys.argv[0])
)


def main():
    # Filter out arguments meant for the HPX runtime
    # argv = [arg for arg in sys.argv[1:] if not arg.startswith("--hpx")]
    argv = [
        arg
        for arg in sys.argv[1:]
        if not arg.startswith("--rate") and not arg.startswith("--hpx")
    ]
    arguments = docopt.docopt(usage, argv)

    dx = float(arguments["<dx>"])
    assert dx > 0, "dx must be strictly positive"

    dy = float(arguments["<dy>"])
    assert dy > 0, "dy must be strictly positive"

    dt = float(arguments["<dt>"])
    assert dt > 0, "dt must be strictly positive"

    T_surf_file = arguments["<T_surf_file>"]

    T_initial_file = arguments["<T_initial_file>"]

    h_total_file = arguments["<h_total_file>"]

    mu_soil_initial_file = arguments["<mu_soil_initial_file>"]

    mu_soil_surf_file = arguments["<mu_soil_surf_file>"]

    U_x_surf_file = arguments["<U_x_surf_file>"]
    U_x_initial_file = arguments["<U_x_initial_file>"]
    gama_soil_surf_file = arguments["<gama_soil_surf_file>"]
    gama_soil_initial_file = arguments["<gama_soil_initial_file>"]
    phase_state_surf_file = arguments["<phase_state_surf_file>"]
    phase_state_initial_file = arguments["<phase_state_initial_file>"]
    thermal_diffusivity_coeff_surf_file = arguments[
        "<thermal_diffusivity_coeff_surf_file>"
    ]
    vegetation_vol_fraction_surf_file = arguments["<vegetation_vol_fraction_surf_file>"]
    vegetation_vol_fraction_initial_file = arguments[
        "<vegetation_vol_fraction_initial_file>"
    ]

    nu_x = float(arguments["<nu_x>"])
    assert nu_x >= 0, "nu_x must be ..."

    nu_y = float(arguments["<nu_y>"])
    assert nu_y >= 0, "nu_y must be ..."

    partition_extent = int(arguments["<partition_extent>"])
    assert partition_extent > 0, "partition_extent must be strictly positive"
    partition_shape = 2 * (partition_extent,)

    time_end = int(arguments["<time_end>"])
    assert time_end >= 0, "time_end must be positive"

    nr_time_steps = round(time_end / dt) + 1

    results_pathname = arguments["<results_pathname>"]
    assert not os.path.splitext(results_pathname)[1]

    # rho_s = 2650
    # porosity = 0.4

    solifluction_simulate(
        dx,
        dy,
        dt,
        T_surf_file,
        T_initial_file,
        h_total_file,
        mu_soil_initial_file,
        mu_soil_surf_file,
        U_x_surf_file,
        U_x_initial_file,
        gama_soil_surf_file,
        gama_soil_initial_file,
        phase_state_surf_file,
        phase_state_initial_file,
        thermal_diffusivity_coeff_surf_file,
        vegetation_vol_fraction_surf_file,
        vegetation_vol_fraction_initial_file,
        nu_x,
        nu_y,
        nr_time_steps,
        partition_shape,
        results_pathname,
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
