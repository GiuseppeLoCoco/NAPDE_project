import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
from firedrake import VTKFile
from firedrake import CheckpointFile
import numpy as np
import argparse
from domain_settings import *
from matplotlib import pyplot as plt
from math import cos, pi as PI

from obstacles import circleObstacle, lineObstacle, rotatingLineObstacle
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function, sqrt, inner
from firedrake.pyplot import triplot, tripcolor, quiver


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

        # start total timer
        t_start = time()

        ########## DATA AND SOLVE
        tol = 1e-10

        # Create mesh
        Lx, Ly = 3, 1
        x_obs = 0.5
        y_obs = 0.5 * Ly
        n = 70
        r_obs = 0.1
        eps = 8.0 / n

        if self.conforming:
            # Test using conforming mesh only with cylinder
            mesh = conforming_mesh(Lx, Ly, x_obs, y_obs, r_obs, n)
            self.obstacle = circleObstacle(y_obs, y_obs, r_obs)
        else:
            mesh = RectangleMesh(n, n // 3, Lx, Ly)
            # Define the obstacle (Using rotatingLine as in the original RIIS code)
            # obstacle = rotatingLineObstacle(x_obs, 0.0, x_obs, y_obs, eps)
            self.obstacle = circleObstacle(y_obs, y_obs, r_obs)

        
        # Data
        T_end = 1.0            # final time
        num_steps = 20    # number of time steps
        dt = T_end / num_steps # time step size
        mu = 0.1         # dynamic viscosity
        rho = 1            # density

        # RIIS Penalty Parameters
        if self.conforming:
            R = 0.0
        else:
            R = 1000.0

        f = Constant((0, 0))
        t = Constant(0.0)

        x, y = SpatialCoordinate(mesh)
        inflow_profile = as_vector(((1.0-exp(-t)) * 4.0*y*(1.0 - y), 0.0))

        
        # Define boundaries
        inflow_id = 1
        outflow_id = 2
        walls_ids = (3, 4)

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
        phi_expr = self.obstacle.distExpr(mesh, t)
        delta_expr = self.obstacle.delta(mesh, t)
        us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))

        if self.conforming and self.moving:
            w = us_expr
        else:
            w = Constant((0.0, 0.0))

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
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

        if self.conforming:
            basedir = 'cyl/'+dir1+dir2+'n'+str(n)+'/'
        else:
            basedir = 'cyl/'+dir1+dir2+'n'+str(n)+'_R'+str(R)+'/'

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        basedir_vel = basedir + 'velocity/'
        basedir_pres = basedir + 'pressure/'
        basedir_delta = basedir + 'delta/'
        basedir_phi = basedir + 'phi/'
        basedir_mesh = basedir + 'mesh/'

        subdirs = [basedir_vel, basedir_pres, basedir_delta, basedir_phi, basedir_mesh]

        for folder in subdirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Create VTK files for visualization output
        xdmffile_u = VTKFile(basedir+'velocity.pvd')
        xdmffile_p = VTKFile(basedir+'pressure.pvd')
        xdmffile_phi = VTKFile(basedir+'phi.pvd')
        xdmffile_delta = VTKFile(basedir+'delta.pvd')
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

        if self.conforming and self.moving:
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

            current_us_x = float(assemble(self.obstacle.us_x(t) * dx(domain=mesh)) / assemble(Constant(1.0) * dx(domain=mesh)))

            # Current Position of the cylinder
            amplitude = 12 * r_obs
            displ_x = amplitude * 0.5 * (1 - cos(0.2 * PI * t))
            xc = self.obstacle.x_obs + displ_x
            yc = self.obstacle.y_obs

            if self.conforming and self.moving:
                pass

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Save solution to file (VTK/PVD)
            phiFun.interpolate(phi_expr)
            deltaFun.interpolate(delta_expr)
            saveVTK(file_dict, t_val, uh, ph, phiFun, deltaFun)

            # Update previous solution
            uh_n.assign(uh)

            # Save the results in a checkpoint file for post-processing
            if self.conforming and self.moving:
                with CheckpointFile(basedir_mesh + 'mesh_t={:.2f}.h5'.format(t_val), 'w') as chk:
                    chk.save_mesh(mesh)
                
            with CheckpointFile(basedir_vel + 'velocity_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(uh, name='velocity')

            with CheckpointFile(basedir_pres + 'pressure_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(ph, name='pressure')

            with CheckpointFile(basedir_phi + 'phi_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(phiFun, name='phi')

            with CheckpointFile(basedir_delta + 'delta_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(deltaFun, name='delta')

            self.plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)


    def plot_results(self, mesh, uh, ph, t_val, basedir):

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        time = f" a t = {t_val:.2f}" if t_val is not None else ""

        # ==========================================
        # MESH PLOT
        # ==========================================

        axes[0].set_title(f"Mesh{time}")
        triplot(mesh, axes=axes[0], interior_kw={"color": "k", "linewidth": 0.5})
        axes[0].set_aspect('equal')


        # ==========================================
        # PRESSURE PLOT
        # ==========================================

        axes[1].set_title(f"Pressure (p){time}")
        # tripcolor mappa i valori scalari in colori
        plot_p = tripcolor(ph, axes=axes[1], cmap='coolwarm')
        fig.colorbar(plot_p, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        axes[1].set_aspect('equal')


        # ==========================================
        # VELOCITY PLOT
        # ==========================================

        axes[2].set_title(f"Velocity (u){time}")
        V_scalar = FunctionSpace(mesh, "CG", 1)
        u_mag = Function(V_scalar)
        u_mag.interpolate(sqrt(inner(uh, uh)))

        plot_u = tripcolor(u_mag, axes=axes[2], cmap='viridis')
        fig.colorbar(plot_u, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        
        """
        V_cg1_vec = VectorFunctionSpace(mesh, "CG", 1)
        uh_cg1 = Function(V_cg1_vec).interpolate(uh)
        
        x_coords = mesh.coordinates.dat.data_ro[:, 0]
        y_coords = mesh.coordinates.dat.data_ro[:, 1]

        U_vel = uh_cg1.dat.data_ro[:, 0]
        V_vel = uh_cg1.dat.data_ro[:, 1]
        """

        coarse_mesh = RectangleMesh(24, 8, 3, 1) 
        V_coarse = VectorFunctionSpace(coarse_mesh, "CG", 1)

        uh_coarse = Function(V_coarse).interpolate(uh, allow_missing_dofs=True)

        x_coarse = coarse_mesh.coordinates.dat.data_ro[:, 0]
        y_coarse = coarse_mesh.coordinates.dat.data_ro[:, 1]
        U_coarse = uh_coarse.dat.data_ro[:, 0]
        V_coarse = uh_coarse.dat.data_ro[:, 1]

        axes[2].quiver(x_coarse, y_coarse, U_coarse, V_coarse, 
                       color='black', scale=40, width=0.001, headwidth=2, pivot='mid')

        axes[2].set_aspect('equal')

        plt.tight_layout()

        plot_dir = basedir + 'plots/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_dir + f'plot_t={t_val:.2f}.png', dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes RIIS solver script')
    args = parser.parse_args()
    
    # Istanziamo la classe e chiamiamo il solver
    solver = RIIS_solver(conforming=False, moving=True)
    solver.RIIS_solve(args)
