from matplotlib import pyplot as plt
import numpy as np
from math import *

def phistar(r):
    if abs(r) < 0.5:
        return 3.0/4.0 - r**2
    elif abs(r) >= 0.5 and abs(r) <= 1.5:
         return 9.0/8.0 - 3.0/2.0*abs(r) + r*r/2.0
    else:
        return 0
# def phistar(r):
#     if abs(r) < 1:
#         return 1-abs(r)
#     else:
#         return 0
    
L = 1
# def funS(x):
#     if abs(x)>L:
#         return 0
#     else:
#         return 0.7+np.sin(0.7*pi/L*(x-0.25*L))
def fun(x):
    return 0.4+np.sin(0.7*pi/2*(x-0.25))

def interp(fun, PP, xxS):
    ell = L/3
    return [np.sum([fun(P)*phistar((P-xS)/ell) for P in PP]) for xS in xxS], \
            [[fun(P)*phistar((P-xS)/(ell)) for xS in xxS] for P in PP]

xx = np.linspace(-2,2,100)
yy = [fun(x) for x in xx]
xxS = [x for x in xx if abs(x)<=L]
yyS = [fun(x) for x in xxS]

PP4 = np.linspace(-L,L,4)
Iv4, phis = interp(fun, PP4, xxS)
PP8 = np.linspace(-L,L,10)
Iv8, phis8 = interp(fun, PP8, xxS)

plt.plot(xx, yy, '-r', label=r'$v:\Omega\to\mathbb{R}$')
plt.plot(PP4, [fun(P) for P in PP4], 'k+')
plt.plot(xxS, Iv4, '-k', label=r'$I_\phi v:\Omega^s\to\mathbb{R}$'+', #P=4')
for phi in phis:
    plt.plot(xxS, phi)
plt.plot(xxS, Iv8, '-b', label=r'$I_\phi v:\Omega^s\to\mathbb{R}$'+', #P=8')
# plt.plot(xx, [phistar(x) for x in xx])
# plt.plot(xx, [phistar((x-3)/0.2) for x in xx],label='aa')
plt.legend()
plt.show()