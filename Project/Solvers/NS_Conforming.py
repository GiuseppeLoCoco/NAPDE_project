import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from domain_settings import *
from math import cos, pi as PI, sin # Keep for local math.cos, math.sin usage

from obstacles import circleObstacle, squareObstacle, rotatingLineObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

class Conforming_solver:
    def __init__(self, moving=True):
        self.moving = moving

    def conforming_solve(self, args=None):

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10

        # Lx, Ly, x_obs, y_obs, n, r_obs are imported from user_inputs
        # For demonstration, let's assume a default obstacle type.
        # In a real scenario, this would come from user_inputs.py
        obstacle_type = "rotating_line" # "circle", "square", or "rotating_line"

        y_obs = 0.5 * Ly
        n = 25
        r_obs = 0.1 # Used for circleObstacle
        side_length = 0.2 # Used for squareObstacle
        # For rotatingLineObstacle
        xA_line = Lx / 2 - 0.1
        yA_line = Ly / 2
        xB_line = Lx / 2 + 0.1
        yB_line = Ly / 2
        line_thickness = 0.02 # Small thickness for the line obstacle

        if obstacle_type == "circle":
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
        elif obstacle_type == "square":
            self.obstacle = squareObstacle(x_obs, y_obs, side_length)
        elif obstacle_type == "rotating_line":
            self.obstacle = rotatingLineObstacle(xA_line, yA_line, xB_line, yB_line, thickness=line_thickness)
        else:
            raise ValueError(f"Unsupported obstacle type: {obstacle_type}")

        mesh = conforming_mesh(Lx, Ly, self.obstacle, n)
        
        # Data
        T_end = 10.0            # final time
        num_steps = 20    # number of time steps
        dt = T_end / num_steps # time step size
        mu = 0.1         # dynamic viscosity
        rho = 1            # density
        f = Constant((0, 0)) # Define f here

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
        us_expr = as_vector((self.obstacle.us_x(t_param), self.obstacle.us_y(t_param)))

        if self.moving:
            w = us_expr
        else:
            w = Constant((0.0, 0.0))

        # Create boundary conditions using the dedicated function (t_param is updated via time_varying_bc)
        bcs = create_bcs_conforming(W, mesh, w)

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx + inner(f,v)*dx

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
            # t.assign(t_val) # t is not used directly anymore, t_param is used via time_varying_bc
            time_varying_bc(t_val) # Aggiorna il t_param globale per le BCs

            if self.moving and isinstance(self.obstacle, rotatingLineObstacle):
                print("Re-meshing for rotating obstacle...")
                # 1. Create new mesh for the current time
                new_mesh = conforming_mesh(Lx, Ly, self.obstacle, n, t_val=t_val)

                # 2. Define new function spaces
                V_new = VectorFunctionSpace(new_mesh, "CG", 2)
                Q_new = FunctionSpace(new_mesh, "CG", 1)
                W_new = V_new * Q_new

                # 3. Project old solution onto the new mesh
                uh_n_new = project(uh_n, V_new)
                uh_n = uh_n_new

                # 4. Update mesh and spaces for the current step
                mesh = new_mesh
                V, Q, W = V_new, Q_new, W_new
                u, p = TrialFunctions(W)
                v, q = TestFunctions(W)
                sol = Function(W)
                uh, ph = sol.subfunctions

                # 5. Re-create BCs and variational forms on the new spaces
                bcs = create_bcs_conforming(W, mesh, w)
                a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
                      + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
                      + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
                      - div(v)*p*dx \
                      + div(u)*q*dx
                L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx + inner(f,v)*dx

            else:
                # For non-rotating obstacles, use ALE for translation
                if self.moving:
                    displ_x = self.obstacle.displ_x(t_param)
                    displ_y = self.obstacle.displ_y(t_param)
                    
                    # ALE.move expects a Constant or Expression for displacement
                    # The displacement is relative to the initial position
                    ALE.move(mesh, as_vector((displ_x, displ_y)))

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