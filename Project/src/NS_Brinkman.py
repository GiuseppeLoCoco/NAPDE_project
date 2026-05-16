from firedrake import *
import numpy as np
import os
from matplotlib import pyplot as plt
from math import pi as PI

from obstacles import circleObstacle

CONFORMING = False  # if True, switch off Brinkman term (R=0) and create mesh conforming to obstacle
MOVING = False      # if True, the immersed obstacle is moving (see obstacles.py)

def saveVTK(file_dict, t, uh, ph, phi, chi):
  uh.rename('u','u')
  ph.rename('p','p')
  phi.rename('phi','phi')
  chi.rename('chi','chi')
  file_dict['u'].write(uh, time=t)
  file_dict['p'].write(ph, time=t)
  file_dict['phi'].write(phi, time=t)
  file_dict['chi'].write(chi, time=t)

########## DATA AND SOLVER

tol = 1e-10

# Create mesh
Lx, Ly = 3, 1
y_obs = 0.5
r_obs = 0.1
n = 50

if CONFORMING:
    raise NotImplementedError("CONFORMING=True requires gmsh in Firedrake, not implemented here.")
else:
    mesh = RectangleMesh(n, n//3, Lx, Ly)

# Data
T_end = 10.0            # final time
num_steps = 20    # number of time steps
dt = T_end / num_steps # time step size
mu = 0.1         # dynamic viscosity
rho = 1            # density

f  = Constant((0, 0))
t = Constant(0.0)

# Coordinates for expressions
x, y = SpatialCoordinate(mesh)
inflow_profile = as_vector(((1.0-exp(-t)) * 4.0*y*(1.0 - y), 0.0))

R = n*20
if CONFORMING:
  R = 0

# Define boundaries
# For RectangleMesh: 1: x=0, 2: x=Lx, 3: y=0, 4: y=Ly
inflow_id = 1
outflow_id = 2
walls_ids = (3, 4)

obstacle = circleObstacle(y_obs,y_obs,r_obs)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids)
bcs = [bcu_inflow, bcu_walls]

# Define trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Define functions for solutions at previous and current time steps
uh_n = Function(V)
sol = Function(W)
uh, ph = sol.subfunctions

# Define variational problem
phi_expr = obstacle.distExpr(mesh, t)
chi_expr = obstacle.chi(mesh, t)
us_expr = as_vector((obstacle.us_x(t), obstacle.us_y(t)))

a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
      + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
      - div(v)*p*dx \
      + div(u)*q*dx \
      + Constant(R) * inner(u,v) * chi_expr * dx
L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
      + inner(f,v)*dx \
      + Constant(R) * inner(us_expr,v)* chi_expr * dx

if CONFORMING:
    dir1 = 'conforming/'
else:
    dir1 = 'brinkman/'

if MOVING:
    dir2 = 'moving/'
else:
    dir2 = 'steady/'

basedir = 'cyl/'+dir1+dir2+'n'+str(n)+'_R'+str(R)+'/'
if not os.path.exists(basedir):
    os.makedirs(basedir)

# Create VTK files for visualization output
xdmffile_u = File(basedir+'velocity.pvd')
xdmffile_p = File(basedir+'pressure.pvd')
xdmffile_phi = File(basedir+'phi.pvd')
xdmffile_chi = File(basedir+'chi.pvd')
file_dict = {'u': xdmffile_u, 'p': xdmffile_p, 'phi': xdmffile_phi, 'chi': xdmffile_chi}

# Time-stepping
t_val = 0.0
uh_n.assign(0.0)

DG1 = FunctionSpace(mesh, 'DG', 1)
phiFun = Function(DG1)
chiFun = Function(DG1)
phiFun.interpolate(phi_expr)
chiFun.interpolate(chi_expr)

saveVTK(file_dict, t_val, uh, ph, phiFun, chiFun)

for step in range(num_steps):
    # Update current time
    t_val += dt
    print('t =', t_val)
    t.assign(t_val)

    solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

    # Save solution to file (VTK/PVD)
    phiFun.interpolate(phi_expr)
    chiFun.interpolate(chi_expr)
    saveVTK(file_dict, t_val, uh, ph, phiFun, chiFun)

    # Update previous solution
    uh_n.assign(uh)

    # Print max velocity
    print('\tu max:', uh.dat.data.max())
