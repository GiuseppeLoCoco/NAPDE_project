from firedrake import *
import numpy as np
import os
import argparse
from domain_settings import *
from matplotlib import pyplot as plt
from math import pi as PI

from obstacles import lineObstacle, rotatingLineObstacle

def saveVTK(file_dict, t, uh, ph, phi, delta):
  uh.rename('u','u')
  ph.rename('p','p')
  phi.rename('phi','phi')
  delta.rename('delta','delta')
  file_dict['u'].write(uh, time=t)
  file_dict['p'].write(ph, time=t)
  file_dict['phi'].write(phi, time=t)
  file_dict['delta'].write(delta, time=t)

class RIIS_solver:
    def __init__(self, conforming=False, moving=True):
        self.conforming = conforming
        self.moving = moving

    def RIIS_solve(self, args=None):
        ########## DATA AND SOLVER

        tol = 1e-10

        # Create mesh
        Lx, Ly = 3, 1
        x_obs = 0.5
        y_obs = 0.5*Ly
        n = 200

        if self.conforming:
            raise NotImplementedError("CONFORMING=True requires gmsh in Firedrake, not implemented here.")
        else:
            mesh = RectangleMesh(n, n//3, Lx, Ly)

        # Data
        T_end = 1.0            # final time
        num_steps = 20    # number of time steps
        dt = T_end / num_steps # time step size
        mu = 0.1         # dynamic viscosity
        rho = 1            # density

        f  = Constant((0, 0))
        t = Constant(0.0)

        x, y = SpatialCoordinate(mesh)
        inflow_profile = as_vector(((1.0-exp(-t)) * 4.0*y*(1.0 - y), 0.0))

        R = 1000
        eps = 8/n
        if self.conforming:
          R = 0

        # Define boundaries
        inflow_id = 1
        outflow_id = 2
        walls_ids = (3, 4)

        obstacle = rotatingLineObstacle(y_obs, 0, y_obs, y_obs, eps)

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
        delta_expr = obstacle.delta(mesh, t)
        us_expr = as_vector((obstacle.us_x(t), obstacle.us_y(t)))

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx \
              + Constant(R/eps) * inner(u,v) * delta_expr * dx
        L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
              + inner(f,v)*dx \
              + Constant(R/eps) * inner(us_expr,v) * delta_expr * dx

        if self.conforming:
            dir1 = 'conforming/'
        else:
            dir1 = 'RIIS/'

        if self.moving:
            dir2 = 'moving/'
        else:
            dir2 = 'steady/'

        basedir = 'beam/'+dir1+dir2+'n'+str(n)+'_eps'+str(eps)+'/'
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # Create VTK files for visualization output
        xdmffile_u = File(basedir+'velocity.pvd')
        xdmffile_p = File(basedir+'pressure.pvd')
        xdmffile_phi = File(basedir+'phi.pvd')
        xdmffile_delta = File(basedir+'delta.pvd')
        file_dict = {'u': xdmffile_u, 'p': xdmffile_p, 'phi': xdmffile_phi, 'delta': xdmffile_delta}

        # Time-stepping
        t_val = 0.0
        uh_n.assign(0.0)

        DG1 = FunctionSpace(mesh, 'DG', 1)
        phiFun = Function(DG1)
        deltaFun = Function(DG1)
        phiFun.interpolate(phi_expr)
        deltaFun.interpolate(delta_expr)

        saveVTK(file_dict, t_val, uh, ph, phiFun, deltaFun)

        for step in range(num_steps):
            # Update current time
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Save solution to file (VTK/PVD)
            phiFun.interpolate(phi_expr)
            deltaFun.interpolate(delta_expr)
            saveVTK(file_dict, t_val, uh, ph, phiFun, deltaFun)

            # Update previous solution
            uh_n.assign(uh)

            # Print max velocity
            print('\tu max:', uh.dat.data.max())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes RIIS solver script')
    args = parser.parse_args()
    
    # Istanziamo la classe e chiamiamo il solver
    solver = RIIS_solver_class(conforming=False, moving=True)
    solver.RIIS_solver(args)
