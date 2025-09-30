Governing Equations
===================

This section explains the governing equations used in solifluction.

For simplification it is assumed that :math:`w = 0`.

Momentum equation
------------------------------------------------

Momentum equation in x direction:

.. math::

   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}
   = g \sin(\alpha_x) - \frac{\mu}{\rho_s} \frac{\partial^2 u}{\partial z^2} - \frac{1}{\rho_s} \gamma' \frac{\partial h}{\partial x}

Momentum equation in y direction:

.. math::

   \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}
   = g \sin(\alpha_y) - \frac{\mu}{\rho_s} \frac{\partial^2 v}{\partial z^2} - \frac{1}{\rho_s} \gamma' \frac{\partial h}{\partial y}

where

.. math::

   \gamma' = \frac{\gamma_\text{surf}}{\cos(\alpha)} - \gamma_w \cos(\alpha)

Here:

- :math:`u, v` are the velocity components in x and y directions
- :math:`\alpha_x` and :math:`\alpha_y` are the slope angles in x and y directions
- :math:`\mu` is dynamic viscosity
- :math:`\rho_s` is the soil particle density
- :math:`g` is gravity (9.81 m/s²)
- :math:`h` is the soil surface height
- :math:`\gamma_\text{surf}` and :math:`\gamma_w` are the specific weights (unit weights) of the surface material and water, respectively, defined as :math:`\gamma = \rho g`.

VOF mass conservation (height / volume fraction)
------------------------------------------------

Conservative form:

.. math::

   \frac{\partial h_l}{\partial t} + \frac{\partial (u h_l)}{\partial x} + \frac{\partial (v h_l)}{\partial y} = 0

Here:

- :math:`h_l` is the soil layer thickness
- :math:`u, v` are the horizontal velocity components in x and y directions

Heat transfer
-------------

The temperature :math:`T` evolves according to

.. math::

     \frac{\partial T}{\partial t}=-\frac{1}{\rho c}\frac{\partial (- k \frac{\partial T}{\partial z})}{\partial z}

where:

- :math:`z` is the vertical coordinate (depth) into the ground
- :math:`c` is heat capacity
- :math:`\rho` is soil density
- :math:`k` is the thermal conductivity
