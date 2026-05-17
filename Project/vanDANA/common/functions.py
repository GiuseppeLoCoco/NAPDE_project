from firedrake import *
from scipy.interpolate import splev
import numpy as np
import sys, math, os

# In Firedrake le espressioni analitiche dipendenti dal tempo si gestiscono elegantemente tramite i nodi/coordinate della mesh o classi Python personalizzate passate a Function
class InflowBoundaryValue:
    def __init__(self, param):
        self.param = param
    def __call__(self, x):
        t = self.param["time"].t
        period = self.param["period"]
        nm = self.param["nm"].cycle
        Area = self.param["Area"]
        Vsc = self.param["Vsc"]
        Tsc = self.param["Tsc"]
        func = self.param["func"]
        val = (splev(t*Tsc - nm*period*Tsc, func)/Area)/Vsc
        return val

# Funzione per lo smoothing della mesh adattata a Firedrake (No ALE.move)
def mesh_smoothening(mesh):
    dim = mesh.geometric_dimension()
    x = SpatialCoordinate(mesh)
    
    # In Firedrake, le coordinate sono memorizzate in una funzione vettoriale coordinata
    V = mesh.coordinates.function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    
    smoothing_strength = Constant(1e4)
    dxz = Measure("dx", domain=mesh)
    
    # Formulazione variazionale dello smoothing
    res = (smoothing_strength * inner(grad(u), grad(v)) + dot(u, v)) * dxz
    uh = Function(V)
    
    solve(lhs(res) == rhs(res), uh,
          bcs=[DirichletBC(V, Constant(tuple([0.0]*dim)), "on_boundary")],
          solver_parameters={"ksp_type": "cg", "ksp_rtol": 1e-4})

    # Aggiornamento fisico delle coordinate della mesh (Sostituto di ALE.move)
    mesh.coordinates.assign(mesh.coordinates + uh)
    return mesh, 0.0, 0.0 # radius_ratio non nativo in Firedrake nello stesso modo

def DENO(u, u_components, Mpi, mesh, h_f_X):
    DN_local = 0
    with h_f_X.dat.vec_ro as v_h:
        vertex_values_h_f_X = v_h.array.copy()
    
    vertex_mag_u = np.zeros(len(vertex_values_h_f_X))
    for ui in range(u_components):
        with u[ui].dat.vec_ro as v_u:
            arr_u = v_u.array
        DN_local += np.max(np.abs(arr_u / vertex_values_h_f_X))
        vertex_mag_u += np.square(arr_u)

    NM_local = np.max(np.sqrt(vertex_mag_u) * vertex_values_h_f_X)
    DN = MPI.comm.allreduce(DN_local, op=MPI.MAX)
    NM = MPI.comm.allreduce(NM_local, op=MPI.MAX)
    return DN, NM

def divergence(u, u_components):
    return sum(u[ui].dx(ui) for ui in range(u_components))

def Qf(u, Ec, Re):
    return inner(((2 * Ec) / Re) * sym(grad(u)), grad(u))

def PFE(Tf_n): 
    return conditional(Tf_n >= 0.725, (6.18 * Tf_n * Tf_n) - (7.39 * Tf_n) + 2.21, 0.1)  

def round_decimals_down(number:float, decimals:int=8):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def update_variables(update, u_components, problem_physics):
    u_ = update[0]
    p_ = update[1]
    for ui in range(u_components):
        u_[2][ui].assign(u_[1][ui])
        u_[1][ui].assign(u_[0][ui]) 
    p_[2].assign(p_[1])    
    p_[1].assign(p_[0])
    
    if problem_physics['solve_FSI']:
        Dp_ = update[2]
        Lm_ = update[3]
        Dp_[2].assign(Dp_[1])
        Lm_[1].assign(Lm_[0])
        if problem_physics['solve_temperature']:
            Ts_ = update[5]
            LmTs_ = update[6]    
            Ts_[1].assign(Ts_[0])
            LmTs_[1].assign(LmTs_[0])