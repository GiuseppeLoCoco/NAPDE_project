from firedrake import sqrt, dot, Identity, nabla_grad, inner, outer, Max, conditional, ne

# SUPG/PSPG
def tau(alpha, vel, h, Num, dt):
    vnorm = dot(vel, vel)
    quant = Num * h * h 
    # alpha/sqrt((4.0/dt**2) + (4*(vnorm/h**2)) + (144.0/quant**2))
    tau = alpha / sqrt((4 * (vnorm / (h * h))) + (144.0 / (quant * quant)))   
    return tau

def Pop(u, w):
    return dot(u, nabla_grad(w))

# Crosswind
def tau_cw(C_cw, fx, h, Num, Rs, DOLFIN_EPS=1e-15):
    vnorm = sqrt(inner(nabla_grad(fx), nabla_grad(fx)))
    res_norm = sqrt(dot(Rs, Rs))
    VX = C_cw - ((2 * vnorm) / (h * Num * (res_norm + DOLFIN_EPS)))
    tau_cw = 0.5 * Max(0, VX) * h * (res_norm / (vnorm + DOLFIN_EPS))
    return tau_cw

def D(u): 
    umag = dot(u, u)
    dim = u.geometric_dimension()
    # Proiezione ortogonale alla direzione della linea di flusso
    TX = conditional(ne(umag, 0), Identity(dim) - (outer(u, u) / umag), 0.0 * Identity(dim))    
    return TX      

def Pop_CW(u, fx):
    return dot(D(u), nabla_grad(fx))

# LSIC
def tau_lsic(Re):
    return 2 / (3 * Re)