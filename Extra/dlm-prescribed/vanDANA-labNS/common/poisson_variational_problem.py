from dolfin import *
from ufl import tensors, nabla_div
from .functions import *
from fenicstools import interpolate_nonmatching_mesh
import numpy as np
import sys

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *


class Poisson_problem:

    def __init__(self, input_mesh, bool_stream):
        
        mesh = input_mesh.mesh
        dim = mesh.geometry().dim()
        
        # NB: use the name "velocity" as the main variable to minimize modifications
        V = FunctionSpace(mesh, 'P', fem_degree['velocity_degree'], constrained_domain = constrained_domain)   # main variable
        Z = FunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                            # Lagrange multiplier

        # --------------------------------

        self.u  = TrialFunction(V)
        self.v  = TestFunction(V)
        self.Lm = TrialFunction(Z)

        # Actual solution functions.
        variables = dict();
        u_  = Function(V)
        u_.vector().zero()
        Lm_ = Function(Z)
        Lm_.vector().zero()

        variables.update(u_=u_, Lm_=Lm_)
            
        # --------------------------------

        # Force
        f = Constant(0.0)
        
        # --------------------------------

        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        if dim == 3: self.nz = tensors.unit_vector(2, dim)

        self.A1 = None
        self.null_space = VectorSpaceBasis([])
        self.rs = []
        self.residual = Function(V)
        # @TODO: remove unused matrices
        self.matrix = dict(Mij=None, Kij = None, \
                           Bij = None, \
                           Yij = None, \
                           )
        
        self.f = f
        self.dim = dim
        self.variables = variables
        self.bool_stream = bool_stream
        self.h = CellDiameter(mesh)
        self.h_X = project(self.h, FunctionSpace(mesh, 'P', 1))
        vertex_values_h_X = self.h_X.compute_vertex_values(mesh)
        vn = np.max(1/(vertex_values_h_X))
        self.VN_local = vn*vn
        
        self.F = [V, Z]
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=input_mesh.get_mesh_boundaries())
        self.n = FacetNormal(mesh)
        if dim == 2: self.tang = as_vector([self.n[1], -self.n[0]])

        # --------------------------------

        # Define poisson_solver - NB: use the name "velocity" as the main variable to minimize modifications
        # self.u_solver = PETScKrylovSolver(tentative_velocity_solver['solver_type'], PETScPreconditioner(tentative_velocity_solver['preconditioner_type']))
        self.u_solver = PETScKrylovSolver('gmres', PETScPreconditioner('hypre_amg'))
        self.u_solver.parameters.update(krylov_solvers)
        self.u_solver.set_reuse_preconditioner(True)

        # --------------------------------


    def assemble(self, Lm_):

        d = self.matrix; dim = self.dim; f = self.f
        u = self.u; v = self.v;
        dx = self.dx; ds = self.ds; h = self.h; n = self.n;

        # ----- as flow_variational_problem.pre_assemble()

        d['Mij'] = assemble(u*v * dx, tensor=d['Mij'])                       # Mass matrix 
        d['Kij'] = assemble(dot(nabla_grad(u), nabla_grad(v)) * dx, tensor=d['Kij'])   # Stiffness matrix
        d['Bij'] = assemble(f*v * dx, tensor=d['Bij'])                                  # Body-force vector
        d['Yij'] = assemble(self.Lm*v * dx, tensor=d['Yij'])                           # Lagrange-multiplier matrix

        # ----- as flow_variational_problem.assemble_tentative_velocity()

        A = d['Kij'].copy()

        b = d['Bij'].copy()

        # Lagrange multiplier
        b.axpy(1.0, d['Yij']*Lm_.vector())

        return A, b

    def change_initial_guess(self, u):

        u.vector().zero()
        
    def solve(self, A, x, b, bcs):
        
        [bc.apply(A, b) for bc in bcs]
        self.u_solver.solve(A, x.vector(), b)
        solve(A, x.vector(), b, 'mumps')


class scalar_Lagrange_multiplier_problem:

    # Note to self : This problem is solved on the solid current configuration
    def __init__(self, solid_mesh):

        # --------------------------------

        mesh = solid_mesh.mesh
        dim = mesh.geometry().dim()

        Y = FunctionSpace(mesh, 'P', fem_degree['velocity_degree']) # Data to impose
        Z = FunctionSpace(mesh, 'P', fem_degree['lagrange_degree']) # Lagrange multiplier

        self.Lm = TrialFunction(Z)
        self.e  = TestFunction(Z)
        self.u_trial = TrialFunction(Y)

        variables = dict();
        Lm_ = Function(Z)
        Lm_.vector().zero()
        uf_ = Function(Y)
        uf_.vector().zero()
        variables.update(Lm_=Lm_, uf_=uf_)

        self.F = [Y, Z]
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.variables = variables
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=solid_mesh.get_mesh_boundaries())

        self.matrix = dict(Yij=None, Bij=None)
        
        # self.u_solver = PETScKrylovSolver('bicgstab', PETScPreconditioner('sor'))
        # self.u_solver.parameters.update(krylov_solvers)
        # self.u_solver.set_reuse_preconditioner(True)

        # --------------------------------

    def assemble_lagrange_multiplier(self, rhs_, uf_, rho):
        # assemble problem    Lm*e*dx = (rhs_-rho*uf_)*e*dx

        e = self.e; Lm = self.Lm; dx = self.dx

        self.matrix['Yij'] = assemble(self.Lm*e * dx, tensor=self.matrix['Yij'])
        self.matrix['Bij'] = assemble(Constant(0.0)*e * dx, tensor=self.matrix['Bij'])
        print("SIZES Yij", self.matrix['Yij'].size(0), self.matrix['Yij'].size(1), flush=True)
        A = self.matrix['Yij'].copy()

        b = self.matrix['Bij'].copy()
        b.axpy(1.0, self.matrix['Yij']*rhs_.vector())
        b.axpy(-1.0*rho, self.matrix['Yij']*uf_.vector())

        print("check rhs_:", assemble(rhs_*rhs_*dx))
        print("check uf_:", assemble(uf_*uf_*dx))
        print("check b:", b.norm('l2'))

        return A, b

    def solve_lagrange_multiplier(self, A, x, b):

        # solve(A, x.vector(), b, 'bicgstab', 'sor')
        solve(A, x.vector(), b, 'mumps')
