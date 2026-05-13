from dolfin import *
from ufl import tensors, nabla_div
from .functions import *
from fenicstools import interpolate_nonmatching_mesh
from .constitutive_eq import *
import sys

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *


PI = 3.14159265

# --------------------------------------------------------------------

class PrescribedKinematics:

    #???	# Note to self : This problem is solved on the solid reference configuration
    def __init__(self, solid_mesh):

        mesh = solid_mesh.mesh
        dim = mesh.geometry().dim()

        # --------------------------------

        R  = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])  # Solid displacement
        Z  = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])      # Lagrange multiplier 

        # --------------------------------

        self.test = TestFunction(R) 

        # --------------------------------

        variables = dict();  Dp_=[] 

        for i in range(3):
            Dp_.insert(i, Function(R))  # 0: tot displ; 1: displ increment; 2: old displ increment

        us_ = Function(R)

        variables.update(Dp_=Dp_, us_=us_)

        self.F = [R, Z]
        self.n = FacetNormal(mesh)
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.dim = dim
        self.variables = variables
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=solid_mesh.get_mesh_boundaries())
        
        self.diameter = np.linalg.norm(mesh.coordinates().max(0) - mesh.coordinates().min(0))
        self.amplitude = 0*self.diameter
        self.displ_x = f'({self.amplitude}*0.5*(1-cos(0.4*{PI}*t)))'
        self.displ_y = '0'
        self.displacement = Expression((self.displ_x, self.displ_y), t=0, degree=2)
        self.us_x = f'({self.amplitude}*0.1*{PI}*sin(0.4*{PI}*t))'
        self.us_y = '0'
        self.us = Expression((self.us_x, self.us_y), t=0, degree=2)

        # --------------------------------

    def update_solid(self, mesh, t, dt):

        Dp_ = self.variables['Dp_']; us_ = self.variables['us_']

        self.us.t = self.displacement.t = t
        Dp_[2].vector()[:] = Dp_[1].vector()[:]         # old incr = new incr
        Dp_[1].vector()[:] = -Dp_[0].vector().copy()    # new incr (first part) = -old tot displ
        Dp_[0].interpolate(self.displacement)           # tot displ update
        Dp_[1].vector().axpy(1.0, Dp_[0].vector())      # new incr (second part) += new tot displ

        ALE.move(mesh, Dp_[1])
        mesh.bounding_box_tree().build(mesh)    # needed to update element bounding boxes

        us_.vector().zero()
        us_.vector().axpy(1/float(dt), Dp_[1].vector())


    # Compute drag and lift	(Note to self: written as per 2D cylinder)
    def	post_process_data(self, Mpi, u, p, Dp, t, text_file_handles):

        dx = self.dx; ds = self.ds; n = self.n

        traction = -1*dot(sigma(1, u, p), n)
        drag = 2*assemble(dot(traction, self.nx)*ds)
        lift = 2*assemble(dot(traction, self.ny)*ds)
        jacb = 1.0  # kept for consistency with FSI

        Mpi.set_barrier()
        if Mpi.get_rank() == 0:
            text_file_handles[5].write(f"{t:0,.10G}		{drag:0,.10G}		{lift:0,.10G}		{jacb:0,.10G}\n")  
