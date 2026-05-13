# Import standard modules.
import numpy as np
from matplotlib import pyplot as plt

# Import own modules.
from src.propeller import (
    propeller
)

R = 0.25
N_b = 3
prop = propeller(R, N_b, "input/structure.csv")

N_spanwise = 20
N_chordwise = 60
test = np.linspace(0, 1, N_spanwise)
prop.discretise(N_spanwise, 'cosine', 'linear', N_chordwise, 'linear')


#fig = plt.figure(figsize=(16, 9))
#plt.plot(prop.r, prop.pitch, linestyle='-', marker='x')
#plt.xlabel(r"$r$ (m)")
#plt.ylabel(r"$c$ (m)")
#plt.grid(True)
#plt.show()

fig = plt.figure(figsize=(16, 9))
for idx in range(N_spanwise):
    data = prop.af_r[idx]
    plt.plot(data[0, :], data[1, :], linestyle='-', marker='x')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid(True)
plt.axis('equal')
plt.show()
