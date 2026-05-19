# solver for Yukawa potential
# central field potential - use radial schrodinger eqn

# ========================
# Import Packages/Modules
# ========================
import matplotlib.pyplot as plt           # from hydrogen.py
import numpy as np                        # from hydrogen.py
from jwanglibs import rootfinder as rtf   # from hydrogen.py
import matplotlib.cm as cm                # for colormap plots
from scipy.integrate import trapezoid     # wave function normalization

# For now, we will be using 1.0 for our constants for easy calculation and visualization.

# =================
# Define functions
# =================
# Yukawa Potential
def V(r):
    '''
    if r <= a:
        return -V0                     # potential well inside nuclear radius
    else:
        return np.exp(-lamda*r)/r  # exponential decay outside well
    '''
    return np.where(r <= a, -V0, np.exp(-lamda*r)/r)

# Centrifugal Potential
def centrifugal(r):
    return hbar**2*L*(L+1)/(2*m*r**2)

# Effective Potential
def V_eff(r):                                                  # effective potential
    '''
    from hydrogen.py
    return L*(L+1)/(2*m*r**2) - 1/(4*pi*)/r                    # centrifugal + coulomb
    '''
    return centrifugal(r) + V(r)  # centrifugal + yukawa

# Numerov Schrodinger Equation
def f(r):                            # Sch eqn in Numerov form
    return 2*m*(E-V_eff(r))/hbar**2  #rearrange radial Sch eqn

# Numerov function
def numerov(f, u, n, x, h):     # Numerov integrator for $u''+f(x)u=0$
    nodes, c = 0, h**2/12.       # given $[u_0,u_1]$, return $[u_0,u_1,...,u_{n+1}]$
    f0, f1 = f(x), f(x+h)
    for i in range(n):
        x += h
        f2 = f(x+h)             # Numerov method below, 
        u.append((2*(1-5*c*f1)*u[i+1] - (1+c*f0)*u[i])/(1+c*f2))  # Numerov update
        f0, f1 = f1, f2
        if u[-1]*u[-2] < 0.0:
            nodes += 1
    return u, nodes             # return u, nodes

# Shooting function
def shoot(En):
    global E                    # E needed in f(r)
    E, c, xm = En, (h**2)/6., xL + M*h
    wfup, nup = numerov(f, [0,.1], M, xL, h)  # outward integration from left
    wfdn, ndn = numerov(f, [0,.1], N-M, xR, -h)  # inward integration from right
    dup = ((1+c*f(xm+h))*wfup[-1] - (1+c*f(xm-h))*wfup[-3])/(h+h)
    ddn = ((1+c*f(xm+h))*wfdn[-3] - (1+c*f(xm-h))*wfdn[-1])/(h+h)
    return dup*wfdn[-2] - wfup[-2]*ddn

# ===================
# Initial Conditions
# ===================
xL, xR, N = 10e-6, 12., 3000  # limits, interval
hbar, m = 1.0, 1.0  # constants
V0 = 10.0   # depth of potential well
a = 2.0   # radius of potential well (interaction range)
lamda = 0.2   # yukawa interaction strength
h = (xR-xL)/N   # step size
Lmax, EL = 4, []   # define max L, blank L array
M = int(a*N/xR)   # M = matching point

# ================================================
# Calculate energy for n, l, and associated nodes
# ================================================
Estart, dE = -V0-0.1, 0.001   # scan from well bottom -V0
list_psix = [[] for _ in range(Lmax)]  

for L in range(Lmax):
    n, E1, Ea = L+1, Estart, []
    while (E1 < -4*dE):             # sweep E range for each L for pure bound states (E < 0)
    #while (E1 < np.exp(-lamda*a)/a):   # sweep E range for each L until upper well bound
        E1 += dE
        if (shoot(E1)*shoot(E1 + dE) > 0):
            continue
        E = rtf.bisect(shoot, E1, E1 + dE, 1.e-8)
        Ea.append(E)   # append E value to energy array Ea
        wfup, nup = numerov(f, [0,.1], M-1, xL, h)   # calc wf
        wfdn, ndn = numerov(f, [0,.1], N-M-1, xR, -h)
        psix = np.concatenate((wfup[:-1], wfdn[::-1]))
        psix[M:] *= wfup[-1]/wfdn[-1]   # match
        list_psix[L].append(psix)
        
        print ('nodes = %i, n = %i,l = %i, E = %.8e' %(nup+ndn, n, L, E))
        n += 1
    EL.append(Ea)

# =========
# Plotting
# =========
# Some code for plotting added with help of Claude and ChatGPT

# Create a colormap for different n-values
max_n = max(len(EL[L]) for L in range(Lmax))  # Find max number of n-values
#colors = cm.hsv(np.linspace(0, 1, max_n))

plt.figure()   # plot energy levels
for L in range(Lmax):
    for i in range(len(EL[L])):
        plt.plot([L-.3, L+.3], [EL[L][i]]*2, color='k', #colors[i], 
                 lw=0.8#, label=f'n={i+1}' if L == 0 else ''
                 )

plt.axhline(0, color='k', ls='-', lw=0.5, alpha=1.0)
plt.xlabel('$\ell$'), plt.ylabel('$E$')
plt.ylim(EL[0][0]-0.3, 0.3), plt.xticks(range(Lmax))
plt.savefig('energy.pdf')  # Save the figure
plt.show()

# plot potential
r = np.linspace(xL, xR, N)  # r-range

plt.figure()
plt.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.8)  # solid x-axis
plt.plot(r, V_eff(r), 'r-', lw=1.0, label='$V_{eff}$')
plt.plot(r, V(r), 'g--', lw=1.0, label='$V_{Yuk}$')
plt.plot(r, centrifugal(r), 'b-.', lw=1.0, label='$V_{Cent}$')
plt.xlim(0, 8)
plt.ylim(-V0-0.5, 5)
plt.xlabel('$r$')
plt.ylabel('$V(r)$')
plt.legend()
plt.savefig('potentials.pdf')  # Save the figure
plt.show()

# plot normalized reduced radial wave functions for all L and n values
for L in range(Lmax):
    for i, psi in enumerate(list_psix[L]):
        plt.figure(figsize=(10, 5))
        plt.axhline(0, color='k', ls='-', lw=1.0, alpha=1.0)
        
        psi_truncated = psi[:len(r)]
        norm = np.sqrt(trapezoid(psi_truncated**2, r)) # normalize
        psi_normalized = psi_truncated / norm
        plt.plot(r, psi_normalized, lw=1.5, color='b')
        
        plt.xlabel('$r$')
        plt.ylabel(f'$u_{{{L+i+1}{L}}}(r)$')
        plt.title(f'Reduced Radial Wave Function: $n={L+i+1}$, $\ell = {L}$')
        plt.grid(True, alpha=0.3)
        #plt.xlim(0, xR*1.1)  # Focus on region near potential well
        plt.xlim(0, 6)  # Focus on region near potential well
        #plt.ylim(psi_normalized.min()-0.3,)  # Adjust y-limits to show wave function clearly
        #plt.ylim(psi_normalized[i].min()-0.3, psi_normalized[i].max()+0.3)
        #plt.ylim(-1.5, 1.5)
        plt.tight_layout()
        plt.savefig(f'u{L+i+1}{L}.pdf')  # Save the figure
        plt.show()