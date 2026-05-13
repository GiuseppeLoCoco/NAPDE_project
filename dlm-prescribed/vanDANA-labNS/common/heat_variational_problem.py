from dolfin import *
from ufl import tensors
from .functions import *
from fenicstools import interpolate_nonmatching_mesh
import sys

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *

from math import pi as PI

class Heat_problem:	
                                    
    def __init__(self, background_mesh):
        
        mesh = background_mesh.mesh
        dim = mesh.geometry().dim()

        G  = FunctionSpace(mesh, 'P', fem_degree['temperature_degree'], constrained_domain = heat_constrained_domain)

        # --------------------------------

        self.Tp = TrialFunction(G)
        self.ttf = TestFunction(G)
        
        # variables in the space of the actual solution (background mesh)
        variables = dict(); T_ = []
        
        # T_[0] : current sol of the heat problem
        # T_[1] : old sol of the heat problem
        for i in range(2):
            T_.insert(i, Function(G))

        # delta-interpolation of Lm onto the background domain
        LmBg_ = Function(G)
        LmBg_.vector()[:] = 0.0	
        variables.update(T_=T_, LmBg_=LmBg_)	

        # --------------------------------

        self.A = None
        self.matrix = dict(MTij=None, KTij = None)

        self.F = [G]
        self.dim = dim 
        self.variables = variables
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=background_mesh.get_mesh_boundaries())

        self.h  = CellDiameter(mesh)
        self.n  = FacetNormal(mesh)

        self.heat_mu   = Constant(physical_parameters['heat_mu'])
        self.heat_alpha   = Constant(physical_parameters['heat_alpha'])
        # self.u_ex = Expression('x[0]*x[0]*x[0] - x[1]*x[1]*x[1]', degree=5)
        # self.u_ex = Expression('cos(2*pi*x[0])*cos(2*pi*x[1])', degree=5, pi=PI)
        self.u_ex = Expression('(1-cos(pi*t))*(0.5*(1+cos(pi/(1-0.5)*(sqrt(x[0]*x[0]+x[1]*x[1])-0.5))))', degree=5, pi=PI, t=0)
        self.source_term = Function(G)
        # self.source_term.interpolate(Expression('alpha*(x[0]*x[0]*x[0] - x[1]*x[1]*x[1]) - 6*mu*(x[0]-x[1])', degree=5, alpha=self.heat_alpha, mu=self.heat_mu))
        self.source_term = Expression('(1-cos(pi*t))*('
                                                'alpha/2.0 + (alpha/2.0+pi*pi*mu/(2*0.5*0.5))*cos(pi/(1-0.5)*(sqrt(x[0]*x[0]+x[1]*x[1])-0.5)) '
                                                '+ mu*pi/(2*0.5*sqrt(x[0]*x[0]+x[1]*x[1]))*sin(pi/(1-0.5)*(sqrt(x[0]*x[0]+x[1]*x[1])-0.5))'
                                                ')'
                                                '+pi*sin(pi*t)*(0.5*(1+cos(pi/(1-0.5)*(sqrt(x[0]*x[0]+x[1]*x[1])-0.5))))',
                                                degree=5, alpha=self.heat_alpha, mu=self.heat_mu, pi=PI, t=0)

        # --------------------------------

        # Define problem solver
        self.t_solver = PETScKrylovSolver(energy_conservation_solver['solver_type'], PETScPreconditioner(energy_conservation_solver['preconditioner_type']))
        self.t_solver.parameters.update(krylov_solvers)

        # --------------------------------

    def pre_assemble(self, dt):

        Tp = self.Tp; ttf = self.ttf
        d = self.matrix; mu = self.heat_mu; alpha = self.heat_alpha; dx = self.dx
                
        d['MTij']  = assemble(Tp*ttf*dx, tensor=d['MTij'])
        d['KTij']  = assemble(mu*dot(nabla_grad(Tp), nabla_grad(ttf))*dx + alpha*Tp*ttf*dx, tensor=d['KTij'])
        
        if time_control['adjustable_timestep'] == False:
            self.A = self.matrix['KTij'].copy()
            self.A.axpy(1.0/float(dt), self.matrix['MTij'], True)	# i.e. A = K+1/dt*M

    def assemble_temperature(self, T_, LmBg_, dt):
        # LmBg_ = Lagrange multiplier interpolated onto the background mesh

        d = self.matrix 
        Tp = self.Tp; ttf = self.ttf; h = self.h
        dx = self.dx; ds = self.ds

        if time_control['adjustable_timestep'] == False:
            A = self.A.copy()
        else:	
            print('time adaptivity not implemented')    # implemented in vanDANA for FSI, not here
            exit(1)

        b = self.optimized_rhs(A, T_[1])

        # add Lagrange multiplier term (already interpolated)
        b.axpy(1.0, d['MTij']*LmBg_.vector())	
        
        return A, b
        
    def assemble_temperature_corrector(self, T_, LmBg_, LmBg_old_, dt):
        # dLmBg_ = Lagrange multiplier incrementinterpolated onto the background mesh

        d = self.matrix
        Tp = self.Tp; ttf = self.ttf; h = self.h
        dx = self.dx; ds = self.ds

        if time_control['adjustable_timestep'] == False:
            A = d['MTij'].copy()
            A /= dt
        else:	
            print('time adaptivity not implemented')    # implemented in vanDANA for FSI, not here
            exit(1)

        E1 = A.copy()
        b = E1*T_[0].vector()

        # add Lagrange multiplier term (already interpolated)
        b.axpy(1.0, d['MTij']*LmBg_.vector())	
        b.axpy(-1.0, d['MTij']*LmBg_old_.vector())

        return A, b

    def optimized_rhs(self, LHS_matrix, T_old_):

        E1 = LHS_matrix.copy()
        E1.axpy(-1.0, self.matrix['KTij'], True) # extract mass matrix from backward Euler
        b = E1*T_old_.vector()

        b += assemble(self.source_term*self.ttf*self.dx)
        
        return b

    def solve_temperature(self, A, x, b, bcs):
        
        [bc.apply(A, b) for bc in bcs]
        # self.t_solver.solve(A, x.vector(), b)
        # Alternative: direct solver
        solve(A, x.vector(), b, 'mumps')

    def solve_temperature_corrector(self, A, x, b, bcs):
        
        [bc.apply(A, b) for bc in bcs]
        # self.t_solver.solve(A, x.vector(), b)
        # Alternative: direct solver
        solve(A, x.vector(), b, 'mumps')

    
    # Post-processing functions
    def post_process_data(self, Mpi, T_, t, text_file_handles):
        Tex, chi = heat_create_Tex()
        print('\terror =', sqrt(assemble((T_-Tex)*(T_-Tex)*chi*dx)))

        return