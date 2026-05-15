from __future__ import print_function
from dolfin import *
from mshr import *

import numpy as np
import os
from matplotlib import pyplot as plt
from math import pi as PI

from obstacles import circleObstacle

CONFORMING = False  # if True, switch off Brinkman term (R=0) and create mesh conforming to obstacle
MOVING = False      # if True, the immersed obstacle is moving (see obstacles.py)

def saveXDMF(file_dict, t, uh, ph, phi, chi):
  uh.rename('u','u')
  file_dict['u'].write(uh, t)
  ph.rename('p','p')
  file_dict['p'].write(ph, t)
  phi.rename('phi','phi')
  file_dict['phi'].write(phi, t)
  chi.rename('chi','chi')
  file_dict['chi'].write(chi, t)

########## DATA AND SOLVER

print('pre-set tolerances:', DOLFIN_EPS, DOLFIN_EPS_LARGE)
tol = DOLFIN_EPS_LARGE

# Create mesh
Lx, Ly = 3, 1
y_obs = 0.5
r_obs = 0.1
channel = Rectangle(Point(0, 0), Point(Lx, Ly))
domain = []
if CONFORMING:
  cylinder = Circle(Point(y_obs, y_obs), r_obs)
  domain = channel - cylinder
else:
  domain = channel
n = 50
mesh = generate_mesh(domain, n)
plot(mesh)

# Data
T = 10.0            # final time
num_steps = 20    # number of time steps
dt = T / num_steps # time step size
mu = 0.1         # dynamic viscosity
rho = 1            # density

f  = Constant((0, 0))
inflow_profile = Expression(('(1-exp(-t)) * 4.0*x[1]*(1.0 - x[1])', '0'), degree=2, t=0.0)

R = n*20
if CONFORMING:
  R = 0

# Define boundaries
def inflow(x, on_boundary):
  return on_boundary and near(x[0], 0, tol)
def outflow(x, on_boundary):
  return on_boundary and near(x[0], Lx, tol)
def walls(x, on_boundary):
  return on_boundary and (near(x[1], 0) or near(x[1], Ly))
def cylinder(x, on_boundary):
  return on_boundary and x[0]>y_obs-r_obs-tol and x[0]<y_obs+r_obs+tol and x[1]>y_obs-r_obs-tol and x[1]<y_obs+r_obs+tol

obstacle = circleObstacle(y_obs,y_obs,r_obs)
print(obstacle.distStr())

# Define function spaces
Vel = VectorElement('P', mesh.ufl_cell(), 2)
Qel = FiniteElement('P', mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([Vel,Qel]))
V = W.sub(0).collapse()   # velocity space for "uh_n"

# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow)
bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
bcs = []
if CONFORMING:
  bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
  bcs = [bcu_inflow, bcu_walls, bcu_cylinder]
else:
  bcs = [bcu_inflow, bcu_walls]

# Define trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Define functions for solutions at previous and current time steps
uh_n = Function(V)
sol = Function(W)
uh, ph = sol.split()

# Define variational problem
phi = obstacle.distExpr()
chi = obstacle.chi()
us = Expression((obstacle.us_x, obstacle.us_y), degree=3, t=0)
a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
      + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
      - div(v)*p*dx \
      + div(u)*q*dx \
      + Constant(R) * inner(u,v) * chi * dx
L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
      + inner(f,v)*dx \
      + Constant(R) * inner(us,v)* chi * dx

# Parameters for linear solver
print(LinearVariationalSolver.default_parameters().str(True))


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

# Create XDMF files for visualization output
xdmffile_u = XDMFFile(basedir+'velocity.xdmf')
xdmffile_p = XDMFFile(basedir+'pressure.xdmf')
xdmffile_phi = XDMFFile(basedir+'phi.xdmf')
xdmffile_chi = XDMFFile(basedir+'chi.xdmf')
file_dict = {'u': xdmffile_u, 'p': xdmffile_p, 'phi': xdmffile_phi, 'chi': xdmffile_chi}

File(basedir+'mesh.xml.gz') << mesh

# Create progress bar
# progress = Progress('Time-stepping', num_steps)
# set_log_level(LogLevel.PROGRESS)

# Time-stepping
t = 0
uh_n.interpolate(Constant((0,0)))
phi.t = chi.t = t
phiFun = interpolate(phi,FunctionSpace(mesh,'DG',1))
chiFun = interpolate(chi,FunctionSpace(mesh,'DG',1))
saveXDMF(file_dict, t, uh, ph, phiFun, chiFun)
for n in range(num_steps):

    # Update current time
    t += dt
    print('t =', t)
    inflow_profile.t = t
    if MOVING:
      phi.t = chi.t = t
      us.t = t

    # solve(a == L, sol, bcs=bcs, solver_parameters={'linear_solver': 'gmres',
    #                                                'preconditioner':'petsc_amg',
    #                                                'krylov_solver':{'relative_tolerance':1e-8,
    #                                                                 'error_on_nonconvergence':False}})
    # solve(a == L, sol, bcs=bcs, solver_parameters={'linear_solver': 'gmres',
    #                                                'preconditioner':'jacobi',
    #                                                'krylov_solver':{'relative_tolerance':1e-8,
    #                                                                 'error_on_nonconvergence':False}})
    solve(a == L, sol, bcs=bcs, solver_parameters={'linear_solver': 'umfpack'})

    # Plot solution
    plot(uh, title='Velocity')
    plot(ph, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    phiFun = interpolate(phi,FunctionSpace(mesh,'DG',1))
    chiFun = interpolate(chi,FunctionSpace(mesh,'DG',1))
    saveXDMF(file_dict, t, uh, ph, phiFun, chiFun)

    # Update previous solution
    uh_n, _ = sol.split(deepcopy=True)

    # Update progress bar
    # progress += 1
    print('\tu max:', uh.vector().max())

