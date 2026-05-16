# Import standard modules.
import numpy as np
from matplotlib import pyplot as plt

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

R = 0.15
N_b = 2
prop = propeller(R, N_b, "input/structure.csv")

N_spanwise = 100
N_chordwise = 40
test = np.linspace(0, 1, N_spanwise)
prop.discretise(N_spanwise, 'cosine', 'linear', N_chordwise, 'cubic_spline', False, 4e-4)

mu = 18.03e-6               # Pa s (air)
#mu = 1.0518e-3              # Pa s (water)
rho = 1.225                 # kg/m^3 (air)
#rho = 1000                  # kg/m^3 (water)
c_sound = 343               # m/s (air)
#c_sound = 1482              # m/s (water)
Omega = 3356/60*2*np.pi     # rad/s angular velocity
#Omega = 3000/60*2*np.pi
n = Omega/(2*np.pi)           # rps
J = 1e-3                       # advance ratio
#V = J*n*2*R                    # m/s inflow velocity at infinity
V = 1e-3                    # m/s inflow velocity at infinity
feather = 0  # degrees
analysis = bemt(prop, V, Omega, feather, mu, rho, c_sound)
analysis.generate_polars(-10, 15, 1, 5)
#analysis.generate_polars_simple()
analysis.method_ning()

rho_m = 1200
analysis_2 = dynamics(prop, analysis, rho_m)
max_stress = np.max(analysis_2.normal_stress)
permissible_stress = 56.6*9.81/0.004**2     # N/m^2, https://www.mytechfun.com/pla/prusa

sigma = np.trapezoid(prop.c, prop.r)*prop.N_b/(np.pi*prop.R**2)
print(f"Solidity: {sigma*100:.3f} %")
thrust = np.trapezoid(analysis.dTdr, prop.r)
print(f"Thrust: {thrust:.3f} N")
torque = np.trapezoid(analysis.dDdr*prop.r, prop.r)
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
print(f"Maximum Tension: {np.max(analysis_2.tension):.3f} N")
print(f"Maximum Stress: {max_stress:.3f} N/m^2")
print(f"Permissible Stress: {permissible_stress:.3f} N/m^2")
print(f"Safety Factor: {permissible_stress/max_stress:.3f}")
print(f"Maximum Shear: {np.max(analysis_2.shear):.3f} N")
print(f"Maximum Shear Stress: {np.max(analysis_2.shear_stress):.3f} N/m^2")

#fig = plt.figure(figsize=(16, 9))
#plt.plot(prop.r, prop.pitch, linestyle='-', marker='x')
#plt.xlabel(r"$r$ (m)")
#plt.ylabel(r"$c$ (m)")
#plt.grid(True)
#plt.show()

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
    data = analysis.af_polars_extrap[idx]
    plt.plot(data[:, 0], data[:, 1], linestyle='-')
plt.xlabel(r"$alpha (deg)$")
plt.ylabel(r"$Cl$")
plt.grid(True)
plt.xlim(-90, 90)
fig.savefig("output/Cl.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = analysis.af_polars_extrap[idx]
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
#plt.show()
fig.savefig("output/chord_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, np.rad2deg(prop.pitch), linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$pitch (deg)$")
plt.grid(True)
#plt.show()
fig.savefig("output/pitch_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis.dTdr, linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$dT/dr (N/m)$")
plt.grid(True)
#plt.show()
fig.savefig("output/thrust_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')


fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis.dDdr, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$dD/dr (N/m)$")
plt.grid(True)
#plt.show()
fig.savefig("output/drag_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, np.rad2deg(analysis.alpha_sol), linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$alpha$ (deg)")
plt.grid(True)
#plt.show()
fig.savefig("output/alpha_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(analysis_2.x_blade, analysis_2.y_blade, linestyle='-', marker='x')
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.grid(True)
plt.axis('equal')
fig.savefig("output/blade_geometry_1.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(analysis_2.x_blade, analysis_2.z_blade, linestyle='-', marker='x')
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$z$ (m)")
plt.grid(True)
plt.axis('equal')
fig.savefig("output/blade_geometry_2.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis_2.tension, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$tension$ (N)")
plt.grid(True)
fig.savefig("output/tension.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis_2.normal_stress, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$normal stress$ (N/m^2)")
plt.grid(True)
fig.savefig("output/normal_stress.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis_2.shear, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$shear$ (N)")
plt.grid(True)
fig.savefig("output/shear.svg", format="svg",
            transparent=True, bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, analysis_2.shear_stress, linestyle='-', marker='x')
plt.xlabel(r"$r$ (m)")
plt.ylabel(r"$shear stress$ (N/m^2)")
plt.grid(True)
fig.savefig("output/shear_stress.svg", format="svg",
            transparent=True, bbox_inches='tight')
