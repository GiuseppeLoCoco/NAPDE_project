from __future__ import print_function
from dolfin import *
from mshr import *

import numpy as np
import os
from matplotlib import pyplot as plt
from math import pi as PI

from obstacles import lineObstacle, rotatingLineObstacle

CONFORMING = False  # if True, switch off RIIS term (R=0) and create mesh conforming to obstacle
MOVING = True      # if True, the immersed obstacle is moving (see obstacles.py)

def saveXDMF(file_dict, t, uh, ph, phi, delta):
  uh.rename('u','u')
  file_dict['u'].write(uh, t)
  ph.rename('p','p')
  file_dict['p'].write(ph, t)
  phi.rename('phi','phi')
  file_dict['phi'].write(phi, t)
  delta.rename('delta','delta')
  file_dict['delta'].write(delta, t)

########## DATA AND SOLVER

print('pre-set tolerances:', DOLFIN_EPS, DOLFIN_EPS_LARGE)
tol = DOLFIN_EPS_LARGE

# Create mesh
Lx, Ly = 3, 1
x_obs = 0.5
y_obs = 0.5*Ly
channel = Rectangle(Point(0, 0), Point(Lx, Ly))
domain = []
if CONFORMING:
  triangle = Polygon([Point(y_obs+tol, 0), Point(y_obs, y_obs), Point(y_obs-tol, 0)])
  domain = channel - triangle
else:
  domain = channel
n = 200
mesh = generate_mesh(domain, n)
plot(mesh)

# Data
T = 1.0            # final time
num_steps = 20    # number of time steps
dt = T / num_steps # time step size
mu = 0.1         # dynamic viscosity
rho = 1            # density

f  = Constant((0, 0))
inflow_profile = Expression(('(1-exp(-t)) * 4.0*x[1]*(1.0 - x[1])', '0'), degree=2, t=0.0)

R = 1000
eps = 8/n
if CONFORMING:
  R = 0

# Define boundaries
def inflow(x, on_boundary):
  return on_boundary and near(x[0], 0, tol)
def outflow(x, on_boundary):
  return on_boundary and near(x[0], Lx, tol)
def walls(x, on_boundary):
  return on_boundary and (near(x[1], 0) or near(x[1], Ly))
def triangle(x, on_boundary):
  return on_boundary and x[0]>y_obs-2*tol and x[0]<y_obs+2*tol and x[1]<y_obs+2*tol



#obstacle = lineObstacle(y_obs,0, y_obs,y_obs, eps)
obstacle = rotatingLineObstacle(y_obs,0, y_obs,y_obs, eps)
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
  bcu_triangle = DirichletBC(W.sub(0), Constant((0, 0)), triangle)
  bcs = [bcu_inflow, bcu_walls, bcu_triangle]
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
delta = obstacle.delta()
us = Expression((obstacle.us_x, obstacle.us_y), degree=3, t=0)
a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
      + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
      - div(v)*p*dx \
      + div(u)*q*dx \
      + Constant(R/eps) * inner(u,v) * delta * dx
L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
      + inner(f,v)*dx \
      + Constant(R/eps) * inner(us,v) * delta * dx

# Parameters for linear solver
print(LinearVariationalSolver.default_parameters().str(True))

if CONFORMING:
    dir1 = 'conforming/'
else:
    dir1 = 'RIIS/'

if MOVING:
    dir2 = 'moving/'
else:
    dir2 = 'steady/'

basedir = 'beam/'+dir1+dir2+'n'+str(n)+'_eps'+str(eps)+'/'
if not os.path.exists(basedir):
    os.makedirs(basedir)

# Create XDMF files for visualization output
xdmffile_u = XDMFFile(basedir+'velocity.xdmf')
xdmffile_p = XDMFFile(basedir+'pressure.xdmf')
xdmffile_phi = XDMFFile(basedir+'phi.xdmf')
xdmffile_delta = XDMFFile(basedir+'delta.xdmf')
file_dict = {'u': xdmffile_u, 'p': xdmffile_p, 'phi': xdmffile_phi, 'delta': xdmffile_delta}

File(basedir+'mesh.xml.gz') << mesh

# Create progress bar
# progress = Progress('Time-stepping', num_steps)
# set_log_level(LogLevel.PROGRESS)

# Time-stepping
t = 0
uh_n.interpolate(Constant((0,0)))
phi.t = delta.t = t
phiFun = interpolate(phi,FunctionSpace(mesh,'DG',1))
deltaFun = interpolate(delta,FunctionSpace(mesh,'DG',1))
saveXDMF(file_dict, t, uh, ph, phiFun, deltaFun)
for n in range(num_steps):

    # Update current time
    t += dt
    print('t =', t)
    inflow_profile.t = t
    if MOVING:
      phi.t = delta.t = t
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
    deltaFun = interpolate(delta,FunctionSpace(mesh,'DG',1))
    saveXDMF(file_dict, t, uh, ph, phiFun, deltaFun)

    # Update previous solution
    uh_n, _ = sol.split(deepcopy=True)

    # Update progress bar
    # progress += 1
    print('\tu max:', uh.vector().max())

