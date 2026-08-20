import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from math import cos, pi as PI

from user_inputs import *
from domain_settings import create_brinkman_riis_bcs, time_varying_bc
from obstacles import circleObstacle
from post_processing import save_VTK, create_output_folders, plot_results, save_checkpoint
from Solvers.Stokes_RIIS import solve_stokes_riis_initial

class RIIS_solver:
    def __init__(self, moving=True):
        self.moving = moving

    def RIIS_solve(self, args=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, u_init=None, dt=None, t_final=None):

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10
        eps = 8.0 / n

        mesh = RectangleMesh(n, n // (Lx/Ly), Lx, Ly)

        # Define the obstacle
        self.obstacle = circleObstacle(y_obs, y_obs, r_obs)
        
        # Data
        T_end = 10.0               # final time
        num_steps = 20             # number of time steps
        dt = T_end / num_steps     # time step size
        mu = 0.1                   # dynamic viscosity
        rho = 1                    # density

        # RIIS Penalty Parameters
        R = 1000.0

        f = Constant((0, 0))
        t = Constant(0.0)

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        # Define boundary conditions
        bcs = create_brinkman_riis_bcs(W, mesh)

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

        params = {
            'moving': self.moving,
            'unsteady': True, # RIIS is always unsteady in this setup
            'symmetric': False, # Not applicable but needed for path
            'n': n,
            'R': R
        }
        basedir, file_dict = create_output_folders('RIIS', params, extra_fields=['phi', 'delta'])

        # Time-stepping
        t_val = 0.0
        time_varying_bc(0.0)

        # Initial condition initialization for velocity at t = 0
        if u_init is not None:
            if callable(u_init):
                uh_n.interpolate(u_init(mesh))
            else:
                uh_n.assign(u_init)
        else:
            print("Initializing velocity with stationary Stokes RIIS solver (t=0)...")
            uh_stokes, _ = solve_stokes_riis_initial(
                mesh=mesh, W=W, obstacle=self.obstacle,
                type_obstacle="circle", n=n, R=R,
                f_custom=f_custom, u_exact=u_exact, p_exact=p_exact, g_custom=g_custom
            )
            uh_n.assign(uh_stokes)

        uh.assign(uh_n)

        DG1 = FunctionSpace(mesh, 'DG', 1)
        phiFun = Function(DG1)
        deltaFun = Function(DG1)
        phiFun.interpolate(phi_expr)
        deltaFun.interpolate(delta_expr)

        save_VTK(file_dict, t_val, uh, ph, phi=phiFun, delta=deltaFun)

        for step in range(num_steps):

            # Update current time
            t_val = (step + 1) * dt
            print('t =', t_val)
            t.assign(t_val)
            time_varying_bc(t_val)

            current_us_x = float(assemble(self.obstacle.us_x(t) * dx(domain=mesh)) / assemble(Constant(1.0) * dx(domain=mesh)))

            # Current Position of the cylinder
            amplitude = 12 * r_obs
            displ_x = amplitude * 0.5 * (1 - cos(0.2 * PI * t))
            xc = self.obstacle.x_obs + displ_x
            yc = self.obstacle.y_obs

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Save solution to file (VTK/PVD)
            phiFun.interpolate(phi_expr)
            deltaFun.interpolate(delta_expr)
            save_VTK(file_dict, t_val, uh, ph, phi=phiFun, delta=deltaFun)

            # Update previous solution
            uh_n.assign(uh)

            save_checkpoint(basedir, t_val, mesh=None, moving=self.moving, velocity=uh, pressure=ph, phi=phiFun, delta=deltaFun)
            plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes RIIS solver script')
    args = parser.parse_args()
    
    # Istanziamo la classe e chiamiamo il solver
    solver = RIIS_solver(moving=True)
    solver.RIIS_solve(args)
