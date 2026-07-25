import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from domain_settings import *
from math import cos, pi as PI

from obstacles import circleObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

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
        bcu_wall_bottom = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[0]) # ID 3
        bcu_wall_top = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[1])    # ID 4
        bcu_obstacle = DirichletBC(W.sub(0), w, obstacle_id)
        bcs = [bcu_inflow, bcu_wall_bottom, bcu_wall_top, bcu_obstacle]

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx \
              + inner(f,v)*dx

        params = {
            'moving': self.moving,
            'unsteady': True, # Conforming is always unsteady in this setup
            'symmetric': False, # Not applicable but needed for path
            'n': n
        }
        basedir, file_dict = create_output_folders('conforming', params)

        # Time-stepping
        t_val = 0.0
        uh_n.assign(0.0)

        save_VTK(file_dict, t_val, uh, ph)
        save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph)

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
            save_VTK(file_dict, t_val, uh, ph)

            # Update previous solution
            uh_n.assign(uh)

            save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph)
            plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes Conforming solver script')
    args = parser.parse_args()
    
    solver = Conforming_solver(moving=True)
    solver.conforming_solve(args)