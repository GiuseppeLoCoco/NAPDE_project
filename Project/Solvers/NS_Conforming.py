import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from domain_settings import *
from user_inputs import *
from math import cos, pi as PI, sin # Keep for local math.cos, math.sin usage

from obstacles import circleObstacle, squareObstacle, rotatingLineObstacle, lineObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

class Conforming_solver:
    def __init__(self, moving=True, type_obstacle="cylinder"):

        self.moving = moving
        self.symmetric = True
        self.unsteady = False
        self.instationary = True
        self.mean = True
        self.type_obstacle = type_obstacle

    def conforming_solve(self, args=None):

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10

        if self.type_obstacle == "circle":
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
        elif self.type_obstacle == "square":
            self.obstacle = squareObstacle(x_obs, y_obs, side_length)
        elif self.type_obstacle == "line":
                    self.obstacle = lineObstacle(xA, yA, xB, yB, line_thickness)
        elif self.type_obstacle == "rotating_line":
            self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, line_thickness)
        else:
            raise ValueError(f"Unsupported obstacle type: {self.type_obstacle}")

        mesh = conforming_mesh(Lx, Ly, self.obstacle, n)
        
        tol = 1e-10
        T_end = 10.0             # final time
        num_steps = 20           # number of time steps
        dt = T_end / num_steps   # time step size

        # Dynamic viscosity
        if self.unsteady:
            mu = 0.0015
        else:
            mu = 0.0070

        # Density   
        rho = 1.0  

        # Characteristic velocity
        u_max = 1.0                 # max velocity
        u_mean = 0.667              # mean velocity

        # Choose between u_max and u_mean for the charateristic velocity
        # in order to compute the Reynolds number
        if self.mean:
            u_char = u_mean
        else:
            u_char = u_max

        L_char = self.obstacle.get_characteristic_length()

        # Reynolds number
        Re = rho * L_char * u_char / mu 
        print("\nCharacteristic length L_char = {}".format(L_char))
        print("\nReynolds number Re = {} computed with u_characteristic = {}\n".format(Re, u_char))

        if Re > 80:
            print("\nReynolds number Re = {} --> Unsteady Regime\n", format(Re))
            self.unsteady = True
        else:
            print("\nReynolds number Re = {} --> Steady Regime\n", format(Re))
            self.unsteady = False

        f = Constant((0.0, 0.0))
        t = Constant(0.0)

        # --------------------------------
        # Initialize Flow Variational Problem
        # --------------------------------

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

        # ------- Setup output folders -------
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'unsteady': self.unsteady,
            'symmetric': self.symmetric, 
            'n': n,
            'Re': Re,
        }
        basedir, file_dict = create_output_folders('Conforming', params)

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

            if self.moving:
                print("Re-meshing for moving obstacle...")
                # Create new mesh for the current time
                new_mesh = conforming_mesh(Lx, Ly, self.obstacle, n, t_val=t_val)

                # Define new function spaces
                V_new = VectorFunctionSpace(new_mesh, "CG", 2)
                Q_new = FunctionSpace(new_mesh, "CG", 1)
                W_new = V_new * Q_new

                # Project old solution onto the new mesh
                uh_n_new = Function(V_new, name="Velocity_old")
                uh_n_new.interpolate(uh_n) 
                uh_n = uh_n_new

                # Update mesh and spaces for the current step
                mesh = new_mesh
                V, Q, W = V_new, Q_new, W_new
                u, p = TrialFunctions(W)
                v, q = TestFunctions(W)
                sol = Function(W)
                uh, ph = sol.subfunctions

                # Re-create BCs and variational forms on the new spaces
                bcs = create_bcs_conforming(W, mesh, w)

                a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
                      + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
                      + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
                      - div(v)*p*dx \
                      + div(u)*q*dx
                L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx + inner(f,v)*dx

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Update previous solution
            uh_n.assign(uh)
            
            # Save solution to file (VTK/PVD)
            save_VTK(file_dict, t_val, uh, ph)

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