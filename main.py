# Import standard modules.
import numpy as np
from matplotlib import pyplot as plt
from ambiance import Atmosphere

# Import own modules.
from src.propeller import (
    propeller
)
from src.bemt import (
    bemt
)
from src.dynamics import (
    dynamics
)
from src.creategeometry import (
    create_geometry
)
from src.optimiser import (
    propeller_optimiser
)

""" GENERAL PROPERTIES """
N_spanwise = 20
N_chordwise = 20
chordwise_interpolation = 'cubic_spline'
# chordwise_interpolation = 'linear'
simple_polars = False
viscous = False
optimisation = True
geometry = True
approx_curves = False
N_U = 100
N_W = 50

""" GEOMETRICAL PROPERTIES """
filename = "simple_prop.csv"
R = 0.075
N_b = 3
r_R_0 = 0.2
turbine_airfoil = False
enforce_te_thickness = False
min_te_thickness = 1e-3
round_te = False
left_handed = False

""" FLUID PROPERTIES """
altitude = 0                            # m
atmo_data = Atmosphere(altitude)
# rho = atmo_data.density[0]              # kg/m^3 (air)
# c_sound = atmo_data.speed_of_sound[0]   # m/s (air)
# mu = atmo_data.dynamic_viscosity[0]     # Pa s (air)
mu = 1.0518e-3                        # Pa s (water)
rho = 1000                            # kg/m^3 (water)
c_sound = 1482                        # m/s (water)

""" XFOIL PROPERTIES """
N_crit = 3
alpha_min = -15
alpha_max = 15
dalpha = 1

""" MATERIAL PROPERTIES """
rho_m = 1200                                # kg/m^3
permissible_stress = 56.6*9.81/0.004**2     # N/m^2
# source: https://www.mytechfun.com/pla/prusa

""" OPERATIONAL PROPERTIES """
Omega = 1500/60*2*np.pi     # rad/s angular velocity
V = 2.5                       # m/s inflow velocity at infinity
feather = 0                 # degrees
#design_value = 66
design_value = 400
mode = 'power'

n = Omega/(2*np.pi)         # rps
J = V/(n*2*R)               # advance ratio
# V = J*n*2*R                    # m/s inflow velocity at infinity

""" CALCULATION """
prop = propeller(R, N_b, f"input/{filename}")
prop.discretise(
    r_R_0,
    N_spanwise, 'cosine', 'linear',
    N_chordwise, chordwise_interpolation,
    enforce_te_thickness, min_te_thickness
    )
aero_analysis = bemt(prop, V, Omega, feather, mu, rho, c_sound)
if simple_polars:
    aero_analysis.generate_polars_simple()
else:
    aero_analysis.generate_polars(
        alpha_min, alpha_max, dalpha, N_crit, turbine_airfoil, viscous
        )

if optimisation:
    propeller_optimiser(prop, aero_analysis, mode, design_value)
aero_analysis.chord = prop.c
aero_analysis.pitch = prop.pitch
aero_analysis.method_ning()

dyn_analysis = dynamics(prop, aero_analysis, rho_m)
max_stress = np.max(dyn_analysis.normal_stress)

sigma = np.trapezoid(prop.c, prop.r)*prop.N_b/(np.pi*prop.R**2)
print(f"Solidity: {sigma*100:.3f} %")
thrust = np.trapezoid(aero_analysis.dTdr, prop.r)
print(f"Thrust: {thrust:.3f} N")
torque = np.trapezoid(aero_analysis.dDdr*prop.r, prop.r)
print(f"Torque: {torque:.3f} Nm")
power = torque*Omega
print(f"Power: {power:.3f} W")
C_T = thrust/(rho*np.pi*prop.R**4*Omega**2)
C_T_uiuc = thrust/(rho*n**2*(2*prop.R)**4)
print(f"Thrust Coefficient: {C_T:.3f}")
print(f"UIUC Thrust Coefficient: {C_T_uiuc:.3f}")
C_P = power/(rho*np.pi*prop.R**5*Omega**3)
C_P_uiuc = power/(rho*n**3*(2*prop.R)**5)
print(f"Power Coefficient: {C_P:.3f}")
print(f"UIUC Power Coefficient: {C_P_uiuc:.3f}")
print(f"UIUC eta: {C_T_uiuc*J/C_P_uiuc:.3f}")
FoM = C_T**(1.5)/(C_P*2**0.5)
print(f"Figure of Merit: {FoM:.3f}")
print(f"Maximum Tension: {np.max(dyn_analysis.tension):.3f} N")
print(f"Maximum Stress: {max_stress:.3f} N/m^2")
print(f"Permissible Stress: {permissible_stress:.3f} N/m^2")
print(f"Safety Factor: {permissible_stress/max_stress:.3f}")
print(f"Maximum Shear: {np.max(dyn_analysis.shear):.3f} N")
print(f"Maximum Shear Stress: {np.max(dyn_analysis.shear_stress):.3f} N/m^2")

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = prop.af_r[idx]
    plt.plot(data[0, :], data[1, :], linestyle='-')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(True)
plt.axis('equal')
fig.savefig("output/airfoils.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = aero_analysis.af_polars_extrap[idx]
    plt.plot(data[:, 0], data[:, 1], linestyle='-')
plt.xlabel(r"$alpha (deg)$")
plt.ylabel(r"$Cl$")
plt.grid(True)
plt.xlim(-90, 90)
fig.savefig("output/Cl.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = aero_analysis.af_polars_extrap[idx]
    plt.plot(data[:, 0], data[:, 2], linestyle='-')
plt.xlabel(r"$alpha (deg)$")
plt.ylabel(r"$Cd$")
plt.grid(True)
plt.xlim(-90, 90)
fig.savefig("output/Cd.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, prop.c, linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$chord (m)$")
plt.grid(True)
fig.savefig("output/chord_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, np.rad2deg(prop.pitch), linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$pitch (deg)$")
plt.grid(True)
fig.savefig("output/pitch_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, aero_analysis.dTdr, linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$dT/dr (N/m)$")
plt.grid(True)
plt.title('Thrust Distribution')
fig.savefig("output/thrust_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')


fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, aero_analysis.dDdr, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$dD/dr (N/m)$")
plt.grid(True)
plt.title('Drag Distribution')
fig.savefig("output/drag_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, np.rad2deg(aero_analysis.alpha_sol), linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$alpha$ (deg)")
plt.grid(True)
fig.savefig("output/alpha_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(dyn_analysis.x_blade, dyn_analysis.y_blade, linestyle='-', marker='x')
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.grid(True)
plt.axis('equal')
fig.savefig("output/blade_geometry_1.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(dyn_analysis.x_blade, dyn_analysis.z_blade, linestyle='-', marker='x')
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$z$ (m)")
plt.grid(True)
plt.axis('equal')
fig.savefig("output/blade_geometry_2.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dyn_analysis.tension, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$tension$ (N)")
plt.grid(True)
fig.savefig("output/tension.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dyn_analysis.normal_stress, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$normal stress$ (N/m^2)")
plt.grid(True)
fig.savefig("output/normal_stress.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dyn_analysis.shear, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$shear$ (N)")
plt.grid(True)
fig.savefig("output/shear.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dyn_analysis.shear_stress, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$shear stress$ (N/m^2)")
plt.grid(True)
fig.savefig("output/shear_stress.svg", format="svg",
            transparent=True, bbox_inches='tight')

if geometry:
    create_geometry(
        prop, dyn_analysis, N_U, N_W, left_handed,
        round_te, approximate_curves=approx_curves
        )
