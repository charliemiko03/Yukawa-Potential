import numpy as np
import matplotlib.pyplot as plt

def V(r):
    return np.where(r <= a, -V0, -V0*np.exp(-lamda*r)/r)

def centrifugal(r):
    return hbar**2*L*(L+1)/(2*m*r**2)

def V_eff(r):
    return centrifugal(r) + V(r)

hbar, m, a = 1.0, 1.0, 1.0
V0 = 1.0
lamda = 0.0
r = np.linspace(0.1, 10, 1000)

L = 2.0

plt.figure()
plt.plot(r, centrifugal(r), 'b-', label='Centrifugal')
plt.axhline(0, color='k', linestyle='--')
plt.ylim(-0.5, 0.5)
plt.xlabel('$r$')
plt.ylabel('$V_{eff}(r)$')
plt.legend()
plt.show()

L = 1.0

plt.figure()
plt.plot(r, V(r), 'g-', label='Potential')
plt.axhline(0, color='k', linestyle='--')
plt.ylim(-1.0, 1.0)
plt.xlabel('$r$')
plt.ylabel('$V(r)$')
plt.legend()
plt.show()

L = 1.0

plt.figure()
plt.plot(r, V_eff(r), 'b-', label='$V_{eff}$')
plt.axhline(0, color='k', linestyle='--')
plt.ylim(-1.0, 1.0)
plt.xlabel('$r$')
plt.ylabel('$V_{eff}(r)$')
plt.legend()
plt.show()