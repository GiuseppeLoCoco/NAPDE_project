from firedrake import *
from ufl import tensors, nabla_div, lhs, rhs
import numpy as np
import sys
import inspect

from .functions import *
from .constitutive_eq import *
from .fem_stabilizations import *

sys.path.insert(0, '..')
from user_inputs import *
from utilities.read import *

PI = 3.14159265

class Fluid_problem:	
									
    def __init__(self, fluid_mesh, bool_stream):
        
        allowed_keys = inspect.signature(calc_non_dimensional_numbers).parameters.keys()
        filt_physical_parameters = {k: v for k, v in physical_parameters.items() if k in allowed_keys}
        Re, _, _, Fr = calc_non_dimensional_numbers(**filt_physical_parameters, **characteristic_scales)

        mesh = fluid_mesh.mesh
        dim = mesh.geometric_dimension()
        self.u_components = dim
        
        # Spazi di funzioni in Firedrake
        V = FunctionSpace(mesh, 'P', fem_degree['velocity_degree'])		  
        Q = FunctionSpace(mesh, 'P', fem_degree['pressure_degree'])		  
        Z1 = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                            

        Vp = VectorFunctionSpace(mesh, 'P', fem_degree['velocity_degree'])

        self.u1 = TrialFunction(V)
        self.v = TestFunction(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)
        self.Lm1 = [TrialFunction(Z1.sub(ui)) for ui in range(self.u_components)]

        variables = dict()
        u_ = []
        p_ = []
        for i in range(3):
            u_.insert(i, as_vector([Function(V) for ui in range(self.u_components)]))
            p_.insert(i, Function(Q))

        self.u_inner = as_vector([Function(V) for ui in range(self.u_components)])
        self.p_inner = Function(Q)

        self.p_x = Function(Q)
        self.pvc_factor = 0.0
        if pressure_velocity_coupling == "IPCS": 
            self.pvc_factor = 1.0

        uv = Function(Vp)		
        Lm_f = Function(Z1)
        Lm_f.assign(0.0)

        variables.update(u_=u_, uv=uv, p_=p_, Lm_f=Lm_f)	
            
        vort, psi = Function(Q), Function(Q)
        vort.assign(0.0)
        psi.assign(0.0)
        variables.update(vort=vort, psi=psi)

        f = Constant(tuple([0.0]*dim))
        
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        if dim == 3: 
            self.nz = tensors.unit_vector(2, dim)

        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.matrix = dict(Mij=None, Kij=None, Cij=None,
                           Sij=[None for ui in range(self.u_components)],
                           Bij=[None for ui in range(self.u_components)],
                           Pij=[None for ui in range(self.u_components)],
                           Yij=[None for ui in range(self.u_components)],
                           A1_as1=None, A2_as2=None, b1_Ls1=None, A1_SCW1=None, b2=None)
        
        self.f = f
        self.dim = dim
        self.variables = variables
        self.bool_stream = bool_stream
        
        # Dimensione della cella in Firedrake
        self.h_f = CellVolume(mesh) / FacetArea(mesh) 
        self.h_f_X = project(self.h_f, FunctionSpace(mesh, 'P', 1))
        
        with self.h_f_X.dat.vec_ro as v_idx:
            vn = np.max(1.0 / v_idx.array)
        self.VN_local = vn * vn
        
        self.F = [V, Q, Z1]
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh)	
        self.n = FacetNormal(mesh)
        if dim == 2: 
            self.tang = as_vector([self.n[1], -self.n[0]])

        self.Re = Constant(Re)
        self.Fr = Constant(Fr)

        self.u_ab = as_vector([Function(V) for ui in range(self.u_components)])    
        self.Cij = dot(dot(self.u_ab, nabla_grad(self.u1)), self.v) * self.dx

        # Configurazione Parametri Solutori Krylov (PETSc)
        self.u_solver_params = {
            "ksp_type": tentative_velocity_solver['solver_type'],
            "pc_type": tentative_velocity_solver['preconditioner_type']
        }
        self.p_solver_params = {
            "ksp_type": pressure_correction_solver['solver_type'],
            "pc_type": pressure_correction_solver['preconditioner_type']
        }
        self.u_c_solver_params = {
            "ksp_type": velocity_correction_solver['solver_type'],
            "pc_type": velocity_correction_solver['preconditioner_type']
        }

    def pre_assemble(self, bcs, dt):
        d = self.matrix; Re = self.Re
        u1 = self.u1; v = self.v; p = self.p; q = self.q
        dx = self.dx; f = self.f
        
        d['Mij'] = assemble(dot(u1, v) * dx)
        self.A3 = d['Mij']
        d['Kij'] = assemble(dot((0.5 / Re) * nabla_grad(u1), nabla_grad(v)) * dx)
        
        for ui in range(self.u_components):
            d['Sij'][ui] = assemble(dot(p, v.dx(ui)) * dx)
            d['Bij'][ui] = assemble(dot(f[ui], v) * dx)
            d['Pij'][ui] = assemble(dot(p.dx(ui), v) * dx)
            d['Yij'][ui] = assemble(dot(self.Lm1[ui], v) * dx)

        self.A1 = Matrix(d['Kij'])
        self.A1.axpy(1.0 / float(dt), d['Mij'])
        self.A2 = assemble(dot(nabla_grad(p), nabla_grad(q)) * dx)

    def residual_NS_equation(self, ui, u1, u2, u_ab, p_, Lm_f, f, dt):
        U = 0.5 * (u1 + u2)	
        self.residual = (u1 - u2) / dt + dot(u_ab, nabla_grad(U)) + p_.dx(ui) - nabla_div((2 / self.Re) * nabla_grad(U)) - f[ui] - Lm_f[ui]

    def assemble_tentative_velocity(self, u_, p_, Lm_f, dt):
        d = self.matrix; Re = self.Re; f = self.f
        u1 = self.u1; v = self.v; u_ab = self.u_ab
        dx = self.dx; h_f = self.h_f

        A1 = Matrix(self.A1)

        for ui in range(self.u_components):
            u_ab[ui].assign(0.0)
            u_ab[ui].vector().axpy(1.5, u_[1][ui].vector())
            u_ab[ui].vector().axpy(-0.5, u_[2][ui].vector())

        d['Cij'] = assemble(self.Cij)
        A1.axpy(-0.5, d['Cij'])

        X1 = Matrix(A1)
        X1.axpy(-2.0, d['Kij'])
        b1 = [None] * self.u_components
        for ui in range(self.u_components):
            b1[ui] = Fluid_problem.optimized_rhs(self, ui, X1, u_[1], p_[1])	
        
        A1.axpy(1.0, d['Cij'])

        for ui in range(self.u_components):
            b1[ui].axpy(1.0, d['Yij'][ui] * Lm_f.sub(ui).vector())

        if stabilization_parameters['SUPG_NS']:
            tau_supg = tau(alpha, u_[1], h_f, Re, dt)
            operator_supg = Pop(u_ab, v)
            for ui in range(self.u_components):
                Fluid_problem.residual_NS_equation(self, ui, u1, u_[1][ui], u_ab, p_[1], Lm_f, f, dt)
                S1 = tau_supg * dot(operator_supg, self.residual) * dx
                if ui == 0:
                    d['A1_as1'] = assemble(lhs(S1))
                    A1.axpy(1.0, d['A1_as1'])
                d['b1_Ls1'] = assemble(rhs(S1))
                b1[ui].axpy(1.0, d['b1_Ls1'])

        if stabilization_parameters['crosswind_NS']:
            self.rs.clear()
            for ui in range(self.u_components):
                Fluid_problem.residual_NS_equation(self, ui, u_[1][ui], u_[2][ui], u_ab, p_[1], Lm_f, f, dt)
                self.rs.append(self.residual)
            R = as_vector(self.rs)
            d['A1_SCW1'] = assemble(inner(tau_cw(C_cw, u_[1], h_f, Re, R) * Pop_CW(u_ab, u1), nabla_grad(v)) * dx)
            A1.axpy(1.0, d['A1_SCW1'])

        return A1, b1	        

    def optimized_rhs(self, ui, X1, u, p):
        b = dolfin_to_firedrake_copy(self.matrix['Bij'][ui])
        b.axpy(1.0, X1 * u[ui].vector())
        b.axpy(self.pvc_factor, self.matrix['Sij'][ui] * p.vector())
        return b
	
    def change_initial_guess(self, u):
        for i in range(self.dim):
            u[i].assign(0.0)
		
    def solve_tentative_velocity(self, A, x, b, bcs):
        for ui in range(self.u_components):
            # In Firedrake risolviamo usando LinearSolver o l'operatore solve nativo
            solve(A, x[ui], b[ui], bcs=bcs[ui], solver_parameters=self.u_solver_params)

    def assemble_pressure_correction(self, u_, p_, Lm_f, dt):
        p = self.p; q = self.q; dx = self.dx; f = self.f
        h_f = self.h_f; u_ab = self.u_ab; Re = self.Re; d = self.matrix
	
        L2 = (-1 / dt) * divergence(u_[0], self.u_components) * q * dx
		
        A = Matrix(self.A2)
        if stabilization_parameters['PSPG_NS']:
            tau_pspg = tau(alpha, u_[0], self.h_f, Re, dt)
            operator_pspg = nabla_grad(q)
            self.rs.clear()
            for ui in range(self.u_components):
                Fluid_problem.residual_NS_equation(self, ui, u_[0][ui], u_[1][ui], u_ab, p, Lm_f, f, dt)
                self.rs.append(self.residual)
            R = as_vector(self.rs)	
            S2 = tau_pspg * dot(operator_pspg, R) * dx
            d['A2_as2'] = assemble(lhs(S2))
            A.axpy(1.0, d['A2_as2'])
            L2 += rhs(S2)
		
        b2 = assemble(L2)
        b2.axpy(self.pvc_factor, self.A2 * p_.vector())
        d['b2'] = b2
        return A, b2

    def solve_pressure_correction(self, A, x, b, bcs):
        # La gestione del NullSpace in Firedrake avviene tramite parametri del solver
        params = self.p_solver_params.copy()
        if not bcs:
            params.update({"mat_type": "matfree", "ksp_constant_null_space": True})
        solve(A, x, b, bcs=bcs, solver_parameters=params)
        if not bcs:
            normalize(x)

    def assemble_velocity_correction(self, u_, p_0, p_1, dt):
        b3 = [None] * self.u_components
        self.p_x.assign(0.0)
        self.p_x.vector().axpy(self.pvc_factor, p_1.vector())

        for ui in range(self.u_components):
            b3[ui] = self.matrix['Mij'] * u_[ui].vector()
            b3[ui].axpy(-float(dt), self.matrix['Pij'][ui] * (p_0.vector() - self.p_x.vector())) 
        return b3

    def solve_velocity_correction(self, x, b, bcs):
        A = self.A3
        for ui in range(self.u_components):
            solve(A, x[ui], b[ui], bcs=bcs[ui], solver_parameters=self.u_c_solver_params)

    def assemble_velocity_correction_DLM(self, u_, Lm_, Lm_old_, dt):
        bDLM = [None] * self.u_components
        for ui in range(self.u_components):
            bDLM[ui] = self.matrix['Mij'] * u_[ui].vector()
            bDLM[ui].axpy(float(dt), assemble(Lm_[ui] * self.v * self.dx - Lm_old_[ui] * self.v * self.dx))
        return bDLM

    def post_process_data(self, Mpi, u, p, t, tsp, text_file_handles):
        Re = self.Re; n = self.n; ds = self.ds
        traction = -1 * dot(sigma(Re, u, p), n)
        drag = assemble(dot(traction, self.nx) * ds(4)) / (0.5 * PI / 4)
        lift = assemble(dot(traction, self.ny) * ds(4)) / (0.5 * PI / 4)

        if MPI.comm.rank == 0:
            text_file_handles[0].write(f"{t:0,.10G}		{drag:0,.10G}		{lift:0,.10G}\n")    

    def calc_vorticity_streamfunction(self, u, bcs):
        p = self.p; q = self.q; dx = self.dx
        vort = self.variables['vort']
        psi = self.variables['psi']

        if self.bool_stream:
            a = p * q * dx
            L = (u[0].dx(1) - u[1].dx(0)) * q * dx
            solve(a == L, vort, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

            a = inner(grad(p), grad(q)) * dx
            L = vort * q * dx
            solve(a == L, psi, bcs=bcs, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

        return vort, psi

def dolfin_to_firedrake_copy(vec):
    # Helper per copiare strutture vettoriali simili
    from firedrake.numeric_constants import JustToGetVector
    new_vec = Function(vec.function_space()).vector()
    new_vec.assign(vec)
    return new_vec