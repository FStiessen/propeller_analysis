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

R = 0.15
N_b = 2
prop = propeller(R, N_b, "input/structure.csv")

N_spanwise = 20
N_chordwise = 80
test = np.linspace(0, 1, N_spanwise)
prop.discretise(N_spanwise, 'cosine', 'linear', N_chordwise, 'linear')

mu = 18.03e-6               # Pa s (air)
#mu = 1.0518e-3              # Pa s (water)
rho = 1.225                 # kg/m^3 (air)
#rho = 1000                  # kg/m^3 (water)
V = 1e-3                    # m/s inflow velocity at infinity
c_sound = 343               # m/s (air)
#c_sound = 1482              # m/s (water)
Omega = 3356/60*2*np.pi     # rad/s angular velocity
#Omega = 1000/60*2*np.pi
feather = 0  # degrees
analysis = bemt(prop, V, Omega, feather, mu, rho, c_sound)
analysis.generate_polars(0, 15, 1, 3)
#analysis.generate_polars_simple()
analysis.method_ning()

sigma = np.trapezoid(prop.c, prop.r)*prop.N_b/(np.pi*prop.R**2)
print(f"Solidity: {sigma*100:.3f} %")
dTdr = analysis.c_thrust_sol*prop.r*rho*V**2*np.pi
thrust = np.trapezoid(dTdr, prop.r)
print(f"Thrust: {thrust:.3f} N")
dDdr = analysis.c_drag_sol*prop.r*rho*V**2*np.pi
torque = np.trapezoid(dDdr*prop.r, prop.r)
print(f"Torque: {torque:.3f} Nm")
power = torque*Omega
print(f"Power: {power:.3f} W")
C_T = thrust/(rho*np.pi*prop.R**4*Omega**2)
print(f"Thrust Coefficient: {C_T:.3f}")
C_P = power/(rho*np.pi*prop.R**5*Omega**3)
print(f"Power Coefficient: {C_P:.3f}")
FoM = C_T**(1.5)/(C_P*2**0.5)
print(f"Figure of Merit: {FoM:.3f}")

#fig = plt.figure(figsize=(16, 9))
#plt.plot(prop.r, prop.pitch, linestyle='-', marker='x')
#plt.xlabel(r"$r$ (m)")
#plt.ylabel(r"$c$ (m)")
#plt.grid(True)
#plt.show()

#fig = plt.figure(figsize=(16, 9))
#for idx in range(N_spanwise):
#    data = prop.af_r[idx]
#    plt.plot(data[0, :], data[1, :], linestyle='-', marker='x')
#plt.xlabel(r"$x$")
#plt.ylabel(r"$y$")
#plt.grid(True)
#plt.axis('equal')
#plt.show()

#fig = plt.figure(figsize=(16, 9))
#for idx in range(N_spanwise):
#    data = analysis.af_polars_extrap[idx]
#    plt.plot(data[:, 0], data[:, 1], linestyle='-', marker='x')
#plt.xlabel(r"$alpha$")
#plt.ylabel(r"$Cl$")
#plt.grid(True)
#plt.show()

fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dTdr, linestyle='-', marker='x')
plt.xlabel(r"$r (m)$")
plt.ylabel(r"$dT/dr (N/m)$")
plt.grid(True)
#plt.show()
fig.savefig("output/thrust_distribution.svg", format="svg",
            transparent=True, bbox_inches='tight')


fig = plt.figure(figsize=(16, 9))
plt.plot(prop.r, dDdr, linestyle='-', marker='x')
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
