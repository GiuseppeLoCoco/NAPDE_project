from dolfin import *
from ufl import tensors
from .functions import *
import sys
from math import pi as PI

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *

# Note: Lagrange mulitplier is the fictitious force in both domains 
class Heat_lagrange_multiplier_problem:

    # Note: This problem is solved on the constraint's current configuration
    def __init__(self, constraint_mesh):

        # --------------------------------

        self.mesh = constraint_mesh.mesh     # stored as attribute because possibly moving
        dim = self.mesh.geometry().dim()

        Y = FunctionSpace(self.mesh, 'P', fem_degree['temperature_degree']) # Temperature
        Z = FunctionSpace(self.mesh, 'P', fem_degree['lagrange_degree'])    # Lagrange multiplier

        self.Lm = TrialFunction(Z)
        self.e  = TestFunction(Z)

        variables = dict(); Lm_ = []

        # temperature and data in the constraint domain
        T_, T_data_ = Function(Y), Function(Y)
        T_data_ = Expression('1-cos(pi*t)', degree=5, pi=PI, t=0)
        
        # Lm_[0] : sol of the Lm problem
        # Lm_[1] : delta-interpolated Lm_[0]
        for i in range(2):
            Lm_.insert(i, Function(Z))
            Lm_[i].vector()[:] = 0.0
        variables.update(Lm_=Lm_, T_=T_)	

        self.T_data_ = T_data_
        self.F = [Y, Z]
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.variables = variables
        self.dx = Measure("dx", domain=self.mesh)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=constraint_meshmesh.get_mesh_boundaries())
        
        # displacement fields for the motion of the constraint domain
        Vdisp  = VectorFunctionSpace(self.mesh, 'P', 1) # p-w linear to keep straight edges
        self.incr_disp, self.tot_disp, self.old_tot_disp = Function(Vdisp), Function(Vdisp), Function(Vdisp)
        # --------------------------------

    def assemble_lagrange_multiplier(self, Lm_old_, T_, dt):
        # Lm_old_ = [ sol of a previous Lm problem , its delta-interpolation ]

        e = self.e; Lm = self.Lm; dx = self.dx

        A = ...
        b = ...

        return A, b

    def solve_lagrange_multiplier(self, A, x, b):

        # solve(A, x.vector(), b, 'bicgstab', 'sor')
        solve(A, x.vector(), b, 'mumps')

    def constraint_error(self, T_):
        return errornorm(T_, interpolate(self.T_data_, self.F[0]), 'L2')

    def update_displacement(self, time):
        expression_tot_disp = Expression(('3*time','0'), degree=2, time=time)
        self.tot_disp.interpolate(expression_tot_disp)

    def update_position(self, time, old_time=None):
        
        self.old_tot_disp.assign(self.tot_disp)
        
        self.update_displacement(time)
        if old_time == None:
            self.incr_disp.vector()[:] = self.tot_disp.vector()[:]
        else:
            self.incr_disp.vector()[:] = self.tot_disp.vector()[:] - self.old_tot_disp.vector()[:]

        # move mesh and recompute bounding box
        mapping(self.mesh, self.incr_disp)
        return self.incr_disp, self.tot_disp

    # DO NOTHING at the moment
    def	post_process_data(self, Mpi, Lm, uf_, t, dt, text_file_handles):
        return
