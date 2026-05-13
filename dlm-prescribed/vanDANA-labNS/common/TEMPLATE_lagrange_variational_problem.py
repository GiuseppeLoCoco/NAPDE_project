from dolfin import *
from ufl import tensors, nabla_div
from .functions import *
import sys

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *


# Note: Lagrange mulitplier is the fictitious force in both domains 
class Lagrange_multiplier_problem:

	# Note to self : This problem is solved on the solid current configuration
	def __init__(self, solid_mesh):

		# --------------------------------

		mesh = solid_mesh.mesh
		dim = mesh.geometry().dim()

		Y  = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])                                             # Solid displacement
		M  = FunctionSpace(mesh, 'P', fem_degree['pressure_degree'])                                                       # Solid pressure
		Z2 = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                                 # Lagrange multiplier

		self.Lm = TrialFunction(Z2)
		self.e  = TestFunction(Z2)

		variables = dict(); Lm_ = []; uf_ = []

		uf_, us_ = Function(Y), Function(Y)
		us_.vector()[:] = 0.0
		
		for i in range(2):
			Lm_.insert(i, Function(Z2))
			Lm_[i].vector()[:] = 0.0
		variables.update(Lm_=Lm_, uf_=uf_)	

		self.us_ = us_
		self.F = [Y, M, Z2]
		self.nx = tensors.unit_vector(0, dim)
		self.ny = tensors.unit_vector(1, dim)
		self.variables = variables
		self.dx = Measure("dx", domain=mesh)
		self.ds = Measure("ds", domain=mesh, subdomain_data=solid_mesh.get_mesh_boundaries())
		
		# --------------------------------

	def assemble_lagrange_multiplier(self, Lm_, us_, uf_, dt):

		e = self.e; Lm = self.Lm; dx = self.dx

		A = ...
		b = ...

		return A, b

	def solve_lagrange_multiplier(self, A, x, b):

		solve(A, x.vector(), b, 'bicgstab', 'sor')