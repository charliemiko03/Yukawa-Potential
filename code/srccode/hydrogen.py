#
# Program 9.5: Hydrogen atom by Numerov's method (hydrogen.py)
# J Wang, Computational modeling and visualization with Python
#

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from jwanglibs import rootfinder as rtf

def Veff(r):                    # effective potential
    return (L*(L+1)/(2*mass*r)-1)/r
    
def f(r):                       # Sch eqn in Numerov form
    return 2*mass*(E-Veff(r))
    
def numerov(f, u, n, x, h):     # Numerov integrator for $u''+f(x)u=0$
    nodes, c = 0, h*h/12.       # given $[u_0,u_1]$, return $[u_0,u_1,...,u_{n+1}]$
    f0, f1 = 0., f(x+h)
    for i in range(n):
        x += h
        f2 = f(x+h)             # Numerov method below, 
        u.append((2*(1-5*c*f1)*u[i+1] - (1+c*f0)*u[i])/(1+c*f2))  
        f0, f1 = f1, f2
        if (u[-1]*u[-2] < 0.0): nodes += 1
    return u, nodes             # return u, nodes
    
def shoot(En):
    global E                    # E needed in f(r)
    E, c, xm = En, (h*h)/6., xL + M*h
    wfup, nup = numerov(f, [0,.1], M, xL, h)
    wfdn, ndn = numerov(f, [0,.1], N-M, xR, -h)     # $f'$ from 
    dup = ((1+c*f(xm+h))*wfup[-1] - (1+c*f(xm-h))*wfup[-3])/(h+h)
    ddn = ((1+c*f(xm+h))*wfdn[-3] - (1+c*f(xm-h))*wfdn[-1])/(h+h)
    return dup*wfdn[-2] - wfup[-2]*ddn

xL, xR, N = 0., 120., 2200          # limits, intervals
h, mass = (xR-xL)/N, 1.0            # step size, mass
Lmax, EL, M = 6, [], 100            # M = matching point

Estart, dE = -.5/np.arange(1, Lmax+1)**2-.1, 0.001      # $\sim -1/2n^2$
for L in range(Lmax):
    n, E1, Ea = L+1, Estart[L], []
    while (E1 < -4*dE):             # sweep E range for each L
        E1 += dE
        if (shoot(E1)*shoot(E1 + dE) > 0): continue
        E = rtf.bisect(shoot, E1, E1 + dE, 1.e-8)
        Ea.append(E)
        wfup, nup = numerov(f, [0,.1], M-1, xL, h)      # calc wf
        wfdn, ndn = numerov(f, [0,.1], N-M-1, xR, -h)
        psix = np.concatenate((wfup[:-1], wfdn[::-1]))
        psix[M:] *= wfup[-1]/wfdn[-1]                   # match
        print ('nodes, n,l,E=', nup+ndn, n, L, E)
        n += 1
    EL.append(Ea)
    
'''
plt.figure()                        # plot energy levels
for L in range(Lmax):
    for i in range(len(EL[L])):
        plt.plot([L-.3, L+.3], [EL[L][i]]*2, 'k-')
    plt.xlabel('$l$'), plt.ylabel('$E$')
    plt.ylim(-.51, 0), plt.xticks(range(Lmax))

plt.savefig('H_energy.pdf')
plt.show()
'''

max_n = max(len(EL[L]) for L in range(Lmax))  # Find max number of n-values
num_colors = 8  # Number of colors to cycle through
colors = cm.Set2(np.linspace(0, 1, num_colors))  # Colors for cycling
legend = set()  # track which n values have been added to legend

plt.figure()  # plot energy levels
for L in range(Lmax):
    for i in range(len(EL[L])):
        list_n = L+i+1  # quantum number n
        label = f'$n$ = {list_n}' if list_n not in legend else None
        plt.plot([L-.3, L+.3], [EL[L][i]]*2, color=colors[(list_n - 1) % num_colors], lw=0.8, label=label, alpha=1.0)
        legend.add(list_n)  # append n value to prevent duplicate legend entries

#plt.title('Hydrogen Energies')
plt.axhline(0, color='k', ls='-', lw=0.5, alpha=1.0)
plt.xlabel('$\ell$'), plt.ylabel('$E_0$')
plt.ylim(-0.51, 0), plt.xticks(range(Lmax))
#plt.ylim(EL[0][0]-.3, EL[-1][-1]+.3), plt.xticks(range(Lmax))
#plt.legend(title='$n$ values', loc='lower right')
plt.savefig('H_energy.pdf')
plt.show()


# --- Plot effective potential ---
r = np.linspace(0.01, 20, 1000)   # avoid r = 0 (singularity)

plt.figure()

for L in range(Lmax):
    V = (L*(L+1)/(2*mass*r**2) - 1/r)
    plt.plot(r, V, label=f"L={L}")

plt.xlabel("$r$ (a.u.)")
plt.ylabel("$V_{eff} (r)$")
#plt.title("Hydrogen Effective Potential")
plt.axhline(0, color='k', ls='-', lw='0.8', alpha=0.7)
plt.ylim(-1.5, 2)   # adjust as needed
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('H_eff_pot.pdf')
plt.show()