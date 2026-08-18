import sys
import os
import gc 
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from user_inputs import *
import user_inputs.user_parameters as user_parameters
from math import cos, pi as PI, sin # Keep for local math.cos, math.sin usage

from domain_settings.boundary_conditions import create_bcs_conforming, time_varying_bc, t_param
from domain_settings.obstacles import circleObstacle, squareObstacle, rotatingLineObstacle, lineObstacle
from domain_settings.mesh_settings import conforming_mesh
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders

class Conforming_solver:
    def __init__(self, moving=False, type_obstacle="square", n=None, Re=None):

        self.moving = moving
        # self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n_conforming
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6

    def conforming_solve(self, args=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, t_final=None):

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10

        if self.type_obstacle == "cylinder":
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
        elif self.type_obstacle == "square":
            self.obstacle = squareObstacle(x_obs, y_obs, side_length)
        elif self.type_obstacle == "line":
                    self.obstacle = lineObstacle(xA, yA, xB, yB, thickness=line_thickness)
        elif self.type_obstacle == "rotating_line":
            self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, line_thickness)
        else:
            raise ValueError(f"Unsupported obstacle type: {self.type_obstacle}")

        mesh = conforming_mesh(Lx, Ly, self.obstacle, self.n)
        
        tol = 1e-10
        T_end = float(t_final) if t_final is not None else 20.0
        dt = 0.5
        num_steps = max(1, int(round(T_end / dt)))

        # Reynolds number
        Re = self.Re

        # Density   
        rho = 1.0  

        # Characteristic velocity
        u_char = 1              # mean velocity

        # Charateristic length
        L_char = self.obstacle.get_characteristic_length()

        # Dynamic viscosity
        mu = rho * L_char * u_char / Re

        print(f"\nCharacteristic length L_char = {L_char}")
        print(f"\nReynolds number Re = {Re} computed with u_characteristic = {u_char}\n")

        f = f_custom(mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        u_ex_val = u_exact(mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(mesh) if callable(g_custom) else g_custom
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
        if u_ex_val is not None:
            bcs = [
                DirichletBC(W.sub(0), u_ex_val, 1),
                DirichletBC(W.sub(0), u_ex_val, 3),
                DirichletBC(W.sub(0), u_ex_val, 4),
                DirichletBC(W.sub(0), u_ex_val, 5)
            ]
            if p_ex_val is not None and g_ex_val is None:
                bcs.append(DirichletBC(W.sub(1), p_ex_val, 2))
        else:
            bcs = create_bcs_conforming(W, mesh, w, type_obstacle=self.type_obstacle)

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n,v)*dx + inner(f,v)*dx
        if g_ex_val is not None:
            ds_b = Measure("ds", domain=mesh)
            L += inner(g_ex_val, v)*ds_b(2)

        # ------- Setup output folders -------
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'symmetric': self.symmetric, 
            'n': self.n,
            'Re': Re,
            'is_mms': (u_exact is not None),
        }
        basedir, file_dict = create_output_folders('Conforming', params)

        if u_exact is not None:
            # Steady-state non-linear solve via Newton's method for exact MMS validation
            F_mms = Constant(rho)*inner(dot(uh, nabla_grad(uh)), v)*dx \
                  + Constant(mu)*inner(sym(grad(uh)), sym(grad(v)))*dx \
                  - div(v)*ph*dx + div(uh)*q*dx \
                  - inner(f, v)*dx
            if g_ex_val is not None:
                ds_b = Measure("ds", domain=mesh)
                F_mms -= inner(g_ex_val, v)*ds_b(2)

            solve(F_mms == 0, sol, bcs=bcs, solver_parameters={
                'snes_type': 'newtonls',
                'snes_rtol': 1e-8,
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            })
            save_VTK(file_dict, T_end, uh, ph)
            save_checkpoint(basedir, T_end, mesh, self.moving, velocity=uh, pressure=ph)
            wall_time = time() - t_start
            print(f"Total wall time = {wall_time} seconds\n", flush=True)
            return

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
                uh_n_new.interpolate(uh_n, allow_missing_dofs=True)
                uh_n = uh_n_new

                # Update mesh and spaces for the current step
                mesh = new_mesh
                V, Q, W = V_new, Q_new, W_new
                u, p = TrialFunctions(W)
                v, q = TestFunctions(W)
                sol = Function(W)
                uh, ph = sol.subfunctions

                # Re-create BCs and variational forms on the new spaces
                bcs = create_bcs_conforming(W, mesh, w, type_obstacle=self.type_obstacle)

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
            # Cancella gli oggetti pesanti legati alla vecchia mesh
            if self.moving:
                del a, L, bcs, sol
            
            # Forza lo spazzino di Python a liberare fisicamente la RAM
            gc.collect()

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navier-Stokes Conforming solver script')
    args = parser.parse_args()
    
    solver = Conforming_solver(moving=False, type_obstacle="square")
    solver.conforming_solve(args)