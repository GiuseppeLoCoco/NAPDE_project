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
from obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

class Brinkman_solver:

    def __init__(self, moving=True, type_obstacle="cylinder"):

        self.moving = moving
        self.symmetric = True
        self.unsteady = False
        self.instationary = True
        self.mean = True
        self.type_obstacle = type_obstacle

    def Brinkman_solve(self, args=None):

        # start total timer
        t_start = time()

        # ==================================
        # CREATE MESH
        # ==================================

        if not self.type_obstacle == "line" and not self.type_obstacle == "rotating":

            if self.type_obstacle == "cylinder":
                print("\nObstacle: Cylinder")
            elif self.type_obstacle == "square":
                print("\nObstacle: Square")
            
            if x_obs == y_obs:
                print("\nSymmetric configuration: cylinder centered in the channel")
                self.symmetric = True
            else:
                print("\nAsymmetric configuration: cylinder moved higher in the channel")
                self.symmetric = False

        else:

            self.symmetric = False
            
            if self.type_obstacle == "line":
                print("\nObstacle: Line")
            else:
                print("\nObstacle: Rotating Line")

        mesh = RectangleMesh(n, int(n * Ly / Lx), Lx, Ly)

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

        R = 1000.0

        if self.type_obstacle == "square":
            self.obstacle = squareObstacle(x_obs, y_obs, r_obs)
            self.moving = False
        elif self.type_obstacle == "cylinder":
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
        elif self.type_obstacle == "line":
            self.obstacle = lineObstacle(xA, xB, yA, yB)
        elif self.type_obstacle == "rotating":
            self.obstacle = rotatingLineObstacle(xA, xB, yA, yB)

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

        # Define expressions for Brinkman
        phi_expr = self.obstacle.distExpr(mesh, t)
        chi_expr = self.obstacle.chi(mesh, t)
        us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))

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
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'unsteady': self.unsteady,
            'symmetric': self.symmetric,
            'n': n,
            'R': R,
            'Re': Re,
        }
        basedir, file_dict = create_output_folders('Brinkman', params, extra_fields=['phi', 'chi'])

        # Time-stepping
        t_val = 0.0
        uh_n.assign(0.0)

        DG1 = FunctionSpace(mesh, 'DG', 1)
        phiFun = Function(DG1)
        chiFun = Function(DG1)
        phiFun.interpolate(phi_expr)
        chiFun.interpolate(chi_expr)

        save_VTK(file_dict, t_val, uh, ph, phi=phiFun, chi=chiFun)
        
        save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph, phi=phiFun, chi=chiFun)

        for step in range(num_steps):
            # Update current time
            t_val += dt
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
            phiFun.interpolate(phi_expr)
            chiFun.interpolate(chi_expr)
            
            save_VTK(file_dict, t_val, uh, ph, phi=phiFun, chi=chiFun)

            # Update previous solution
            uh_n.assign(uh)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())
            
            save_checkpoint(basedir, t_val, mesh=None, moving=self.moving, velocity=uh, pressure=ph, phi=phiFun, chi=chiFun)
            plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes Brinkman solver script')
    args = parser.parse_args()

    # Istanziamo la classe e chiamiamo il solver
    solver = Brinkman_solver(moving=True)
    solver.Brinkman_solve(args)
