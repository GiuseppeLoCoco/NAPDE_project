from ast import Constant
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
from firedrake import VTKFile
from firedrake import CheckpointFile
from domain_settings import *
from time import time
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from math import cos, pi as PI
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function, sqrt, inner
from firedrake.pyplot import triplot, tripcolor, quiver

from obstacles import circleObstacle

def saveVTK(file_dict, t, uh, ph, phi, chi):
    uh.rename('u','u')
    ph.rename('p','p')
    phi.rename('phi','phi')
    chi.rename('chi','chi')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    file_dict['phi'].write(phi, time=t)
    file_dict['chi'].write(chi, time=t)

class Brinkman_solver:

    def __init__(self, conforming=True, moving=True):

        self.conforming = conforming
        self.moving = moving
        self.symmetric = True
        self.unsteady = False
        self.instationary = True
        self.mean = True

    def Brinkman_solve(self, args=None):

        # start total timer
        t_start = time()

        # ==================================
        # CREATE MESH
        # ==================================

        Lx, Ly = 3, 1
        y_obs = 0.5

        if self.symmetric:
            x_obs = y_obs
            print("\nSymmetric configuration: cylinder centered in the channel")
        else:
            y_obs = 0.48
            x_obs = 0.5
            Lx = 4
            print("\nAsymmetric configuration: cylinder moved higher in the channel")

        r_obs = 0.1
        n = 100

        self.conforming = True;

        if self.conforming:
            mesh = conforming_mesh(Lx, Ly, x_obs, y_obs, r_obs, n)
        else:
            mesh = RectangleMesh(n, n//3, Lx, Ly)


        # ==================================
        # DATA AND SOLVER
        # ==================================

        tol = 1e-10

        T_end = 10.0            # Final time
        num_steps = 20          # Number of time steps
        dt = T_end / num_steps  # Time step size
        
        # Dynamic viscosity
        if self.unsteady:
            mu = 0.0015
        else:
            mu = 0.0070

        # Density
        rho = 1       

        # Characteristic velocity
        u_max = 1.0                 # max velocity
        u_mean = 0.667              # mean velocity

        # Choose between u_max and u_mean for the charateristic velocity
        # in order to compute the Reynolds number
        if self.mean:
            u_char = u_mean
        else:
            u_char = u_max

        Re = rho * (2*r_obs) * u_char / mu 
        print("\nReynolds number Re = {} computed with u_characteristic = {}".format(Re, u_char))

        if Re > 80:
            print("\nReynolds number Re = {} --> Unsteady Regime\n", format(Re))
            self.unsteady = True
        else:
            print("\nReynolds number Re = {} --> Steady Regime\n", format(Re))
            self.unsteady = False

        f  = Constant((0, 0))
        t = Constant(0.0)

        # Coordinates for expressions
        x, y = SpatialCoordinate(mesh)
        inflow_profile = as_vector(((1.0 - exp(-t)) * 4.0*y*(1.0 - y), 0.0)) 

        if self.conforming:
            R = 0
        else:
            R = 1000.0

        self.obstacle = circleObstacle(y_obs, y_obs, r_obs)       

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        # Define Boundaries
        if self.conforming:
            # Allineamento con i tag assegnati dentro conforming_mesh.py
            inflow_id = 3
            outflow_id = 4
            walls_ids = (1, 2)  # Bottom e Top
            cylinder_id = 5     # Il contorno del cilindro conforme
        else:
            # Per RectangleMesh nativa di Firedrake: 1: x=0, 2: x=Lx, 3: y=0, 4: y=Ly
            inflow_id = 1
            outflow_id = 2
            walls_ids = (3, 4)

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

        # Define expressions for Brinkman
        phi_expr = self.obstacle.distExpr(mesh, t)
        chi_expr = self.obstacle.chi(mesh, t)
        us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))

        if self.conforming and self.moving:
            w = us_expr
        else:
            w = Constant((0.0, 0.0))


        # ==================================
        # DEFINE VARIATIONAL PROBLEM
        # ==================================

        a = Constant(rho)/Constant(dt)*inner(u, v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx \
              + Constant(R) * inner(u, v) * chi_expr * dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx \
              + inner(f, v)*dx \
              + Constant(R) * inner(us_expr, v)* chi_expr * dx


        # =========================================
        # Create the folder
        # =========================================

        if self.conforming:
            dir1 = 'conforming/'
        else:
            dir1 = 'brinkman/'

        if self.moving:
            dir2 = 'moving/'
        else:
            dir2 = 'fixed/'

        if self.unsteady:
            dir3 = 'unsteady/'
            if self.symmetric:
                dir4 = 'symmetric/'
            else:
                dir4 = 'asymmetric/'
        else:
            dir3 = 'steady/'

        if self.unsteady:
            if self.conforming:
                basedir = 'cyl/'+dir1+dir2+dir3+dir4+'n'+str(n)+'/'
            else:
                basedir = 'cyl/'+dir1+dir2+dir3+dir4+'n'+str(n)+'_R'+str(R)+'/'
        else:
            if self.conforming:
                basedir = 'cyl/'+dir1+dir2+dir3+'n'+str(n)+'/'
            else:
                basedir = 'cyl/'+dir1+dir2+dir3+'n'+str(n)+'_R'+str(R)+'/'

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        basedir_vel = basedir + 'velocity/'
        basedir_pres = basedir + 'pressure/'
        basedir_chi = basedir + 'chi/'
        basedir_phi = basedir + 'phi/'
        basedir_mesh = basedir + 'mesh/'

        subdirs = [basedir_vel, basedir_pres, basedir_chi, basedir_phi, basedir_mesh]

        for folder in subdirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # =========================================

        # Create VTK files for visualization output
        xdmffile_u = VTKFile(basedir+'velocity.pvd')
        xdmffile_p = VTKFile(basedir+'pressure.pvd')
        xdmffile_phi = VTKFile(basedir+'phi.pvd')
        xdmffile_chi = VTKFile(basedir+'chi.pvd')
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
        
        if self.conforming and self.moving:
            with CheckpointFile(basedir_mesh + 'mesh_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_mesh(mesh)
        else:
            with CheckpointFile(basedir_mesh + 'mesh.h5', 'w') as chk:
                chk.save_mesh(mesh)


        for step in range(num_steps):
            # Update current time
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)

            current_us_x = float(assemble(self.obstacle.us_x(t) * dx(domain=mesh)) / assemble(Constant(1.0) * dx(domain=mesh)))

            # Current Position of the cylinder
            amplitude = 12 * r_obs
            displ_x = amplitude * 0.5 * (1 - cos(0.2 * PI * t))
            xc = self.obstacle.x_obs + displ_x
            yc = self.obstacle.y_obs
    
            if self.conforming and self.moving:
                mesh, uh_n, uh, ph, phiFun, chiFun, bcs = self.solve_conforming_timestep(
                    xc, yc, Lx, Ly, r_obs, n, t, rho, dt, mu, R, 
                    inflow_id, walls_ids, f, us_expr, old_uh_n=uh_n)
            else:
                solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
                phiFun.interpolate(phi_expr)
                chiFun.interpolate(chi_expr)
                uh_n.assign(uh)

            saveVTK(file_dict, t_val, uh, ph, phiFun, chiFun)

            # Update previous solution
            uh_n.assign(uh)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())

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

            with CheckpointFile(basedir_chi + 'chi_t={:.2f}.h5'.format(t_val), 'w') as chk:
                chk.save_function(chiFun, name='chi')

            self.plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)


    def solve_conforming_timestep(self, xc, yc, Lx, Ly, r_obs, n, t, rho, dt, mu, R, inflow_id, walls_ids, f, us_expr, old_uh_n):

        # Updated mesh for the current time step
        mesh = conforming_mesh(Lx, Ly, xc, yc, r_obs, n)
        
        # Update the functional spaces on the new mesh
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q
        DG1 = FunctionSpace(mesh, 'DG', 1)
        
        # Interpolation of the previous velocity solution
        current_uh_n = Function(V)
        if old_uh_n is not None:
            current_uh_n.interpolate(old_uh_n, allow_missing_dofs=True)
        else:
            current_uh_n.assign(0.0)
        
        # Initialize solution functions on the new mesh
        sol = Function(W)
        uh, ph = sol.subfunctions
        
        x, y = SpatialCoordinate(mesh)
        inflow_profile = as_vector(((1.0 - exp(-t)) * 4.0 * y * (1.0 - y), 0.0))
        
        phiFun = Function(DG1)
        chiFun = Function(DG1)
        phi_expr = self.obstacle.distExpr(mesh, t)
        chi_expr = self.obstacle.chi(mesh, t)
        phiFun.interpolate(phi_expr)
        chiFun.interpolate(chi_expr)
        
        w = us_expr if (self.conforming and self.moving) else Constant((0.0, 0.0))
        
        # Boundary conditions on the new mesh
        cylinder_velocity = us_expr if self.moving else Constant((0.0, 0.0))
        bcs = [
            DirichletBC(W.sub(0), inflow_profile, inflow_id),
            DirichletBC(W.sub(0), Constant((0, 0)), walls_ids),
            DirichletBC(W.sub(0), cylinder_velocity, 5) # Tag 5 = Cylinder Boundary
        ]
        
        # Variational problem 
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        
        a = Constant(rho)/Constant(dt)*inner(u, v)*dx \
             + Constant(rho)*inner(dot(current_uh_n - w, nabla_grad(u)), v)*dx \
             + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
             - div(v)*p*dx \
             + div(u)*q*dx \
             + Constant(R) * inner(u, v) * chi_expr * dx
        
        L = Constant(rho)/Constant(dt)*inner(current_uh_n, v)*dx \
             + inner(f, v)*dx \
             + Constant(R) * inner(us_expr, v) * chi_expr * dx

        # Solver
        solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
        
        # Return the new results on the updated mesh
        return mesh, current_uh_n, uh, ph, phiFun, chiFun, bcs
    

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
    parser = argparse.ArgumentParser(description='Stokes Brinkman solver script')
    args = parser.parse_args()

    # Istanziamo la classe e chiamiamo il solver
    solver = Brinkman_solver(conforming=False, moving=True)
    solver.Brinkman_solve(args)



