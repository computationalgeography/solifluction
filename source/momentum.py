import derivatives_discretization as fdm
import lue.framework as lfr
from boundary_condition import boundary_set

# Eq: d_phi/d_t + (u_x * d_phi/d_x) + (u_z * d_phi/d_z) + (nu_x * d2_phi/d_x2) + (nu_z * d2_phi/d_z2) = rhs

# NOTE: boundary_type = (0 for Dirichlet and 1,2,3,4 for Neumann)

#     Dirichlet_boundary_type = 0
#
#     5---------------------boundary_type=4---------------------8
#     |                                                         |
#     |                                                         |
# boundary_type=1           Neumann_boundary                boundary_type=3
#     |                                                         |
#     |                                                         |
#     |                                                         |
#     6---------------------boundary_type=2---------------------7


def momentum(
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

    phi_internal = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_central)
        - ((nu_z * dt) * fdm.d2z_central)
        + rhs,
        0,
    )

    phi_boundary_1 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_forward)
        - ((nu_z * dt) * fdm.d2z_central)
        + rhs,
        0,
    )

    phi_boundary_2 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_central)
        - ((nu_z * dt) * fdm.d2z_forward)
        + rhs,
        0,
    )

    phi_boundary_3 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_backward)
        - ((nu_z * dt) * fdm.d2z_central)
        + rhs,
        0,
    )

    phi_boundary_4 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_central)
        - ((nu_z * dt) * fdm.d2z_backward)
        + rhs,
        0,
    )

    phi_boundary_5 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_forward)
        - ((nu_z * dt) * fdm.d2z_backward)
        + rhs,
        0,
    )

    phi_boundary_6 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_forward)
        - ((nu_z * dt) * fdm.d2z_forward)
        + rhs,
        0,
    )

    phi_boundary_7 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_backward)
        - ((nu_z * dt) * fdm.d2z_forward)
        + rhs,
        0,
    )

    phi_boundary_8 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_backward)
        - ((nu_z * dt) * fdm.d2z_backward)
        + rhs,
        0,
    )

    # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen), now vegetation is ignored in phase_state but it is considered in vegetation_vol_fraction

    phi = lfr.where(
        boundary_type == 1,
        phi_boundary_1,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 2,
        phi_boundary_2,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 3,
        phi_boundary_3,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 4,
        phi_boundary_4,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 5,
        phi_boundary_5,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 6,
        phi_boundary_6,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 7,
        phi_boundary_7,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 8,
        phi_boundary_8,
        phi_internal,
    )

    phi = boundary_set(
        phi,
        boundary_loc,
        boundary_type,
        Dirichlet_boundary_value,
        Neumann_boundary_value,
        dx,
        dz,
    )

    return phi, phi_internal
