import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Yukawa radial solver in natural units: hbar = m = 1
hbar, m = 1.0, 1.0
g, mu = 2.5, 1.0          # potential strength and screening
l = 0                     # angular momentum quantum number
rmin, rmax, N = 1e-3, 20.0, 2000

# Radial grid
r = np.linspace(rmin, rmax, N)
dr = r[1] - r[0]

# Yukawa + centrifugal effective potential
V = -g * np.exp(-mu * r) / r
Veff = V + hbar**2 * l * (l + 1) / (2 * m * r**2)

# Finite-difference Hamiltonian for u(r)
kin = hbar**2 / (2 * m * dr**2)
diag = 2 * kin + Veff
offdiag = -kin * np.ones(N - 1)

# Solve H u = E u
E, U = eigh_tridiagonal(diag, offdiag)

# Keep only bound states (E < 0)
bound = np.where(E < 0)[0]
if len(bound) == 0:
    print("No bound states found for these parameters.")
    raise SystemExit

print("Bound-state energies:")
for n, i in enumerate(bound[:5], start=1):
    print(f"n={n}, l={l}, E = {E[i]:.6f}")

# Plot potential and first few bound states
plt.figure(figsize=(8, 5))
plt.plot(r, Veff, label=r"$V_{\mathrm{eff}}(r)$", linewidth=2)

for n, i in enumerate(bound[:3], start=1):
    u = U[:, i]
    u /= np.sqrt(np.trapezoid(u**2, r))      # normalize u(r)
    R = u / r                            # radial wf R(r)
    plt.plot(r, 0.4 * u + E[i], label=fr"$u_{{{n}{l}}}(r)$ + E_{n}")

plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.xlabel("r")
plt.ylabel("Energy / wavefunction")
plt.title("Yukawa potential: bound states")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: inspect one radial wavefunction explicitly
i = bound[0]
u = U[:, i]
u /= np.sqrt(np.trapezoid(u**2, r))
R = u / r

plt.figure(figsize=(8, 4))
plt.plot(r, R, label=fr"$R_{{1,{l}}}(r)$")
plt.xlabel("r")
plt.ylabel("R(r)")
plt.title("Ground-state radial wavefunction")
plt.legend()
plt.tight_layout()
plt.show()