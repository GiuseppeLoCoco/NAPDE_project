import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from math import cos, pi as PI

from obstacles import circleObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

class Brinkman_solver:

    def __init__(self, moving=True):

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

        R = 1000.0

        self.obstacle = circleObstacle(x_obs, y_obs, r_obs)       

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        inflow_id = 1
        outflow_id = 2
        walls_ids = (3, 4)

        # Define boundary conditions
        bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
        bcu_wall_bottom = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[0]) # ID 3
        bcu_wall_top = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[1])    # ID 4
        bcs = [bcu_inflow, bcu_wall_bottom, bcu_wall_top]

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
            'unsteady': self.unsteady,
            'symmetric': self.symmetric,
            'n': n,
            'R': R
        }
        basedir, file_dict = create_output_folders('brinkman', params, extra_fields=['phi', 'chi'])

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
