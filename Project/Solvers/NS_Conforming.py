import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
from firedrake import VTKFile
from firedrake import CheckpointFile
import numpy as np
import argparse
from domain_settings import *
from matplotlib import pyplot as plt
from math import cos, pi as PI

from obstacles import circleObstacle
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function, sqrt, inner
from firedrake.pyplot import triplot, tripcolor


def saveVTK(file_dict, t, uh, ph):
  uh.rename('u','u')
  ph.rename('p','p')
  file_dict['u'].write(uh, time=t)
  file_dict['p'].write(ph, time=t)


class Conforming_solver:
    def __init__(self, moving=True):
        self.moving = moving

    def conforming_solve(self, args=None):

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10

        # Create mesh
        Lx, Ly = 3, 1
        x_obs = 0.5
        y_obs = 0.5 * Ly
        n = 25
        r_obs = 0.1

        # Test using conforming mesh only with cylinder
        mesh = conforming_mesh(Lx, Ly, x_obs, y_obs, r_obs, n)
        self.obstacle = circleObstacle(y_obs, y_obs, r_obs)
        
        # Data
        T_end = 10.0            # final time
        num_steps = 20    # number of time steps
        dt = T_end / num_steps # time step size
        mu = 0.1         # dynamic viscosity
        rho = 1            # density

        f = Constant((0, 0))
        t = Constant(0.0)

        x, y = SpatialCoordinate(mesh)
        inflow_profile = as_vector(((1.0-exp(-t)) * 4.0*y*(1.0 - y), 0.0))

        
        # Define boundaries
        inflow_id = 1
        outflow_id = 2
        walls_ids = (3, 4)
        obstacle_id = 5

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        # Define trial and test functions
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        
        # Define functions for solutions at previous and current time steps
        uh_n = Function(V)
        sol = Function(W)
        uh, ph = sol.subfunctions

        # Define variational problem
        us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))

        if self.moving:
            w = us_expr
        else:
            w = Constant((0.0, 0.0))

        # Define boundary conditions
        bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
        bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids)
        bcu_obstacle = DirichletBC(W.sub(0), w, obstacle_id)
        bcs = [bcu_inflow, bcu_walls, bcu_obstacle]

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
              + inner(f,v)*dx

        dir1 = 'conforming/'

        if self.moving:
            dir2 = 'moving/'
        else:
            dir2 = 'steady/'

        basedir = 'cyl/'+dir1+dir2+'n'+str(n)+'/'

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        basedir_vel = basedir + 'velocity/'
        basedir_pres = basedir + 'pressure/'
        basedir_mesh = basedir + 'mesh/'

        subdirs = [basedir_vel, basedir_pres, basedir_mesh]

        for folder in subdirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Create VTK files for visualization output
        xdmffile_u = VTKFile(basedir+'velocity.pvd')
        xdmffile_p = VTKFile(basedir+'pressure.pvd')
        file_dict = {'u': xdmffile_u, 'p': xdmffile_p}

        # Time-stepping
        t_val = 0.0
        uh_n.assign(0.0)

        saveVTK(file_dict, t_val, uh, ph)

        if self.moving:
            with CheckpointFile(basedir_mesh + 'mesh_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_mesh(mesh)
        else:
            with CheckpointFile(basedir_mesh + 'mesh.h5', 'w') as chk:
                chk.save_mesh(mesh)

        for step in range(num_steps):

            # Update current time
            t_val = (step + 1) * dt
            print('t =', t_val)
            t.assign(t_val)

            # Current Position of the cylinder
            amplitude = 12 * r_obs
            displ_x = amplitude * 0.5 * (1 - cos(0.2 * PI * t))
            xc = self.obstacle.x_obs + displ_x
            yc = self.obstacle.y_obs

            if self.moving:
                ALE.move(mesh, Constant((xc - self.obstacle.x_obs, 0.0)))

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Save solution to file (VTK/PVD)
            saveVTK(file_dict, t_val, uh, ph)

            # Update previous solution
            uh_n.assign(uh)

            # Save the results in a checkpoint file for post-processing
            if self.moving:
                with CheckpointFile(basedir_mesh + 'mesh_t={:.2f}.h5'.format(t_val), 'w') as chk:
                    chk.save_mesh(mesh)
                
            with CheckpointFile(basedir_vel + 'velocity_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(uh, name='velocity')

            with CheckpointFile(basedir_pres + 'pressure_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(ph, name='pressure')

            self.plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)


    def plot_results(self, mesh, uh, ph, t_val, basedir):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        time = f" a t = {t_val:.2f}" if t_val is not None else ""

        axes[0].set_title(f"Mesh{time}")
        triplot(mesh, axes=axes[0], interior_kw={"color": "k", "linewidth": 0.5})
        axes[0].set_aspect('equal')

        axes[1].set_title(f"Pressure (p){time}")
        plot_p = tripcolor(ph, axes=axes[1], cmap='coolwarm')
        fig.colorbar(plot_p, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        axes[1].set_aspect('equal')

        axes[2].set_title(f"Velocity (u){time}")
        V_scalar = FunctionSpace(mesh, "CG", 1)
        u_mag = Function(V_scalar).interpolate(sqrt(inner(uh, uh)))
        plot_u = tripcolor(u_mag, axes=axes[2], cmap='viridis')
        fig.colorbar(plot_u, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        axes[2].set_aspect('equal')

        plt.tight_layout()

        plot_dir = basedir + 'plots/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_dir + f'plot_t={t_val:.2f}.png', dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes Conforming solver script')
    args = parser.parse_args()
    
    solver = Conforming_solver(moving=True)
    solver.conforming_solve(args)