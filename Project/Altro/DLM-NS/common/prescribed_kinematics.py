from firedrake import *
from ufl import tensors
from .functions import *
from .constitutive_eq import *
import sys

sys.path.insert(0, '..')
from user_inputs import *
from utilities.read import *

PI = 3.14159265

class PrescribedKinematics:
    def __init__(self, solid_mesh):
        mesh = solid_mesh.mesh
        dim = mesh.geometric_dimension()

        R = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])  
        Z = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])      

        self.test = TestFunction(R) 
        variables = dict()
        Dp_ = [Function(R) for _ in range(3)] # 0: tot displ; 1: displ increment; 2: old displ increment

        us_ = Function(R)
        variables.update(Dp_=Dp_, us_=us_)

        self.F = [R, Z]
        self.n = FacetNormal(mesh)
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.dim = dim
        self.variables = variables
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh)
        
        # Calcolo ampiezze geometriche usando coordinate native Firedrake
        coords = mesh.coordinates.dat.vec_ro.array
        self.diameter = np.linalg.norm(coords.max() - coords.min())
        self.amplitude = 0.0 * self.diameter

    def update_solid(self, mesh, t, dt):
        Dp_ = self.variables['Dp_']
        us_ = self.variables['us_']

        Dp_[2].assign(Dp_[1])         
        Dp_[1].assign(-Dp_[0])    
        
        # Sostituzione di Expression con l'estrazione geometrica esplicita in Firedrake
        
        displ_x = (self.amplitude * 0.5 * (1.0 - cos(0.2 * pi * t)))
        displ_y = 0.0
        
        # Interpolazione analitica UFL su spazio discreto
        Dp_[0].interpolate(as_vector([displ_x, displ_y]))
        Dp_[1].vector().axpy(1.0, Dp_[0].vector())      

        # Muoviamo la mesh ridefinendo le sue coordinate (Equivalente Firedrake ad ALE.move)
        mesh.coordinates.assign(mesh.coordinates + Dp_[1])

        us_.assign(0.0)
        us_.vector().axpy(1.0 / float(dt), Dp_[1].vector())

    def post_process_data(self, Mpi, u, p, Dp, t, text_file_handles):
        ds = self.ds; n = self.n
        traction = -1 * dot(sigma(1, u, p), n)
        drag = 2 * assemble(dot(traction, self.nx) * ds)
        lift = 2 * assemble(dot(traction, self.ny) * ds)
        jacb = 1.0  

        if MPI.comm.rank == 0:
            text_file_handles[5].write(f"{t:0,.10G}		{drag:0,.10G}		{lift:0,.10G}		{jacb:0,.10G}\n")