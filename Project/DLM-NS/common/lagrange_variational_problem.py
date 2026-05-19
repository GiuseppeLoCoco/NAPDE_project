from firedrake import *
from ufl import tensors, nabla_div
from .functions import *
import sys

sys.path.insert(0, '..')
from user_inputs import *
from utilities.read import *

class Lagrange_multiplier_problem:
    def __init__(self, solid_mesh):
        mesh = solid_mesh.mesh
        dim = mesh.geometric_dimension()

        Y = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])                                             
        Z2 = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                                 

        self.Lm = TrialFunction(Z2)
        self.e = TestFunction(Z2)

        variables = dict()
        uf_ = Function(Y)
        us_ = Function(Y)
        us_.assign(0.0)
		
        Lm_ = [Function(Z2), Function(Z2)]
        Lm_[0].assign(0.0); Lm_[1].assign(0.0)
        
        variables.update(Lm_=Lm_, uf_=uf_)	

        self.us_ = us_
        self.F = [Y, Z2]
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.variables = variables
        self.dx = Measure("dx", domain=mesh)
		
    def assemble_lagrange_multiplier(self, Lm_, us_, uf_, dt):
        e = self.e; Lm = self.Lm; dx = self.dx
        A = assemble(dot(Lm, e) * dx)
        b = assemble((1 / dt) * dot(us_ - uf_, e) * dx + dot(Lm_[1], e) * dx)
        return A, b

    def solve_lagrange_multiplier(self, A, x, b):
        # Parametri equivalenti a 'bicgstab' + 'sor' in PETSc
        solve(A, x, b, solver_parameters={'ksp_type': 'bicgstab', 'pc_type': 'sor'})


class Solid_temperature_lagrange_multiplier_problem:
    def __init__(self, solid_mesh):
        rho, Spht, K, Ld, Nw, Sm = calc_non_dimensional_solid_properties(**physical_parameters, **characteristic_scales)
        Re, Pr, Ec, Fr = calc_non_dimensional_numbers(**physical_parameters, **characteristic_scales)
        Pe = Re * Pr     
        if not problem_physics['viscous_dissipation']: 
            Ec = 0.0

        mesh = solid_mesh.mesh
        M = FunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])            		
        S = FunctionSpace(mesh, 'P', fem_degree['temperature_degree'])				
		
        self.LmT = TrialFunction(M) 		      
        self.ls = TestFunction(M)			 

        variables = dict()
        Ts_ = [Function(S), Function(S)]
        LmTs_ = [Function(M), Function(M)]
        LmTs_[0].assign(0.0); LmTs_[1].assign(0.0)

        variables.update(Ts_=Ts_, LmTs_=LmTs_)	
		
        self.F = [M, S]
        self.Re = Constant(Re)
        self.Ec = Constant(Ec)
        self.Pe = Constant(Pe)
        self.rho = Constant(rho)
        self.Spht = Constant(Spht)
        self.K = Constant(K)
        self.variables = variables
        self.n = FacetNormal(mesh)
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh)
		
    def assemble_solid_temperature_lagrange_multiplier(self, Ts_, uf_, dt):
        LmT = self.LmT; ls = self.ls; dx = self.dx
        a7 = assemble((-1) * LmT * ls * dx)
        b7 = ((self.rho * self.Spht) - 1) * (1 / dt) * (Ts_[0] - Ts_[1]) * ls * dx + ((0.5 * (self.K - 1)) / self.Pe) * dot(nabla_grad(Ts_[0]) + nabla_grad(Ts_[1]), nabla_grad(ls)) * dx 

        if problem_physics['viscous_dissipation']:
            b7 += Qf(uf_, self.Ec, self.Re) * ls * dx
			
        return a7, assemble(b7)	

    def solve_solid_temperature_lagrange_multiplier(self, A, x, b):
        solve(A, x, b, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})