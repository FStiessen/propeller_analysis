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

R = 0.25
N_b = 3
prop = propeller(R, N_b, "input/structure.csv")

N_spanwise = 4
N_chordwise = 40
test = np.linspace(0, 1, N_spanwise)
prop.discretise(N_spanwise, 'linear', 'linear', N_chordwise, 'linear')

mu = 18.03e-6
rho = 1.225
V = 1
Omega = 3000/60*2*np.pi
analysis = bemt(prop, V, Omega, mu, rho)
analysis.generate_polars(-10, 20, 1, 0.1)

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

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = analysis.af_polars_extrap[idx]
    plt.plot(data[:, 0], data[:, 2], linestyle='-', marker='x')
plt.xlabel(r"$alpha$")
plt.ylabel(r"$Cl$")
plt.grid(True)
plt.show()
