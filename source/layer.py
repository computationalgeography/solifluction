from typing import Any


class Layer:
    def __init__(
        self,
        u_x: Any,
        u_z: Any,
        T: Any,
        h_mesh: Any,
        mu_soil: Any,
        density_soil: Any,
        phase_state: Any,
        k_conductivity_heat: Any,
        rho_c_heat: Any,
        vegetation_vol_fraction: Any,
    ) -> None:

        self.u_x = u_x  # Velocity in x direction
        self.u_z = u_z  # Velocity in z direction - y is almost in the gravity
        # direction normal to bed rock, x and z are layer coordinate parallel to bed rock
        self.T = T  # Temperature
        self.h_mesh = h_mesh  # soil layer thickness in mesh
        self.mu_soil = mu_soil  # soil viscosity
        self.density_soil = density_soil  #  soil density
        self.phase_state = phase_state  # Phase state (fluid, solid (ice), vegetation)
        self.k_conductivity_heat = k_conductivity_heat
        self.rho_c_heat = rho_c_heat
        self.vegetation_vol_fraction = vegetation_vol_fraction
