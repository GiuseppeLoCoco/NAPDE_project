import sys
import os
import gc 
from time import time, perf_counter
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
from post_processing import (
    save_VTK, save_checkpoint, plot_results, create_output_folders,
    find_latest_checkpoint, load_checkpoint_solution, setup_pvd_resume
)
from Solvers.Stokes_solver import solve_stokes_initial

class Conforming_solver:
    def __init__(self, moving=False, type_obstacle="square", n=None, Re=None, structured=False, print_iteration_time=None):

        self.moving = moving
        # self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n_conforming
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.structured = structured
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6
        self.print_iteration_time = print_iteration_time if print_iteration_time is not None else getattr(solver_options, 'print_iteration_time', True)

    def conforming_solve(self, args=None, mesh=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, u_init=None, dt=None, t_final=None, print_iteration_time=None, resume=None):

        if print_iteration_time is not None:
            self.print_iteration_time = print_iteration_time

        if resume is None:
            resume = getattr(solver_options, 'resume_simulation', True)

        # start total timer
        t_start = time()

        # =========== DATA AND SOLVE ===========
        tol = 1e-10

        if obstacle is not None:
            self.obstacle = obstacle
        elif self.type_obstacle in ["none", "None", None]:
            self.obstacle = None
        else:
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

        if mesh is None:
            if self.obstacle is not None:
                mesh = conforming_mesh(Lx, Ly, self.obstacle, self.n, structured=self.structured)
            else:
                if self.structured:
                    mesh = RectangleMesh(self.n, max(4, int(round(self.n * Ly / Lx))), Lx, Ly)
                else:
                    mesh = unstructured_rectangle_mesh(0.0, Lx, 0.0, Ly, self.n, Ly_ref=Ly)
        
        tol = 1e-10
        T_end = float(t_final) if t_final is not None else 5.0
        dt = float(dt) if dt is not None else 0.1
        num_steps = max(1, int(round(T_end / dt)))

        # Reynolds number
        Re = self.Re

        # Density   
        rho = 1.0  

        # Characteristic velocity
        u_char = 1              # mean velocity

        # Characteristic length
        L_char = self.obstacle.get_characteristic_length() if self.obstacle is not None else 1.0

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

        if self.obstacle is not None and hasattr(self.obstacle, 'us_field'):
            us_expr = self.obstacle.us_field(mesh, t_param)
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
            us_expr = as_vector((self.obstacle.us_x(t_param), self.obstacle.us_y(t_param)))
        else:
            us_expr = Constant((0.0, 0.0))


        if self.moving:
            w = us_expr
        else:
            w = Constant((0.0, 0.0))

        # Create boundary conditions using the dedicated function (t_param is updated via time_varying_bc)
        if u_ex_val is not None:
            bcs = [
                DirichletBC(W.sub(0), u_ex_val, 1),
                DirichletBC(W.sub(0), u_ex_val, 3),
                DirichletBC(W.sub(0), u_ex_val, 4)
            ]
            if self.obstacle is not None:
                bcs.append(DirichletBC(W.sub(0), u_ex_val, 5))
            if g_ex_val is None:
                bcs.append(DirichletBC(W.sub(0), u_ex_val, 2))
                if p_ex_val is not None:
                    bcs.append(DirichletBC(W.sub(1), p_ex_val, 2))
        else:
            bcs = create_bcs_conforming(W, mesh, w, type_obstacle=self.type_obstacle)

        a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + 2.0 * Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
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

        # ----------------------------------
        # Checkpoint Discovery & Resume
        # ----------------------------------
        resuming = False
        start_step = 0
        latest_t = None

        if resume:
            latest_t, vel_file, press_file, mesh_file = find_latest_checkpoint(basedir)
            if latest_t is not None:
                if latest_t >= T_end - 1e-9:
                    print(f"\n--- Conforming simulation already completed up to t = {latest_t:.2f}s (target T_end = {T_end:.2f}s) in {basedir} ---", flush=True)
                    mesh_loaded, uh_loaded, ph_loaded = load_checkpoint_solution(vel_file, press_file)
                    return mesh_loaded, uh_loaded, ph_loaded
                else:
                    print(f"\n--- Resuming Conforming simulation from t = {latest_t:.2f}s up to T_end = {T_end:.2f}s (found checkpoint in {basedir}) ---", flush=True)
                    if self.moving:
                        mesh_loaded, uh_loaded, ph_loaded = load_checkpoint_solution(vel_file, press_file)
                        mesh = mesh_loaded
                        V = VectorFunctionSpace(mesh, "CG", 2)
                        Q = FunctionSpace(mesh, "CG", 1)
                        W = V * Q
                        u, p = TrialFunctions(W)
                        v, q = TestFunctions(W)
                        uh_n = Function(V)
                        uh_n.assign(uh_loaded)
                        sol = Function(W)
                        uh, ph = sol.subfunctions
                        uh.assign(uh_n)
                        if ph_loaded is not None:
                            try:
                                ph.assign(ph_loaded)
                            except Exception:
                                ph.interpolate(ph_loaded, allow_missing_dofs=True)
                    else:
                        load_checkpoint_solution(vel_file, press_file, target_u=uh_n, target_p=ph)
                        uh.assign(uh_n)

                    resuming = True
                    setup_pvd_resume(basedir, file_dict, latest_t)

        if resuming:
            t_val = latest_t
            start_step = int(round(latest_t / dt))
            time_varying_bc(latest_t)
        else:
            t_val = 0.0
            time_varying_bc(0.0)

            # Initial condition initialization for velocity at t = 0
            if u_init is not None:
                if callable(u_init):
                    uh_n.interpolate(u_init(mesh))
                else:
                    uh_n.assign(u_init)
            else:
                print("Initializing velocity with stationary Stokes solver (t=0)...", flush=True)
                uh_stokes, _ = solve_stokes_initial(
                    mesh=mesh, bcs=bcs, mu=mu, f_custom=f_custom, g_custom=g_custom, W=W
                )
                uh_n.assign(uh_stokes)

            uh.assign(uh_n)

            save_VTK(file_dict, t_val, uh, ph)
            save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph)

        for step in range(start_step, num_steps):
            t_step_start = perf_counter()

            # Update current time
            t_val = round((step + 1) * dt, 10)
            print('t =', t_val)
            time_varying_bc(t_val)

            if self.moving:
                print("Re-meshing for moving obstacle...")
                # Create new mesh for the current time
                new_mesh = conforming_mesh(Lx, Ly, self.obstacle, self.n, t_val=t_val, structured=self.structured)

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

                # Update obstacle velocity w on the new mesh
                if self.obstacle is not None and hasattr(self.obstacle, 'us_field'):
                    w = self.obstacle.us_field(mesh, t_param)
                elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
                    w = as_vector((self.obstacle.us_x(t_param), self.obstacle.us_y(t_param)))
                else:
                    w = Constant((0.0, 0.0))

                # Re-create BCs and variational forms on the new spaces
                bcs = create_bcs_conforming(W, mesh, w, type_obstacle=self.type_obstacle)

                a = Constant(rho)/Constant(dt)*inner(u,v)*dx \
                      + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
                      + 2.0 * Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
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
            print('\tu_max:', uh.dat.data.max(), flush=True)

            t_step_duration = perf_counter() - t_step_start
            if self.print_iteration_time:
                print(f"\tTempo impiegato per l'iterazione {step + 1}/{num_steps}: {t_step_duration:.4f} s", flush=True)

            # Cancella gli oggetti pesanti legati alla vecchia mesh
            if self.moving:
                del a, L, bcs, sol
            
            # Forza lo spazzino di Python a liberare fisicamente la RAM
            gc.collect()

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)
        return mesh, uh, ph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navier-Stokes Conforming solver script')
    parser.add_argument('--moving', action='store_true', default=True, help='Use moving obstacle')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'square', 'line', 'rotating', 'rotating_line'],
                        help='Type of obstacle to use in the simulation.')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size (default: 0.1)')
    parser.add_argument('--t_final', type=float, default=5.0, help='Final simulation time (default: 5.0)')
    parser.add_argument('--print_time', action='store_true', default=None, help='Print iteration time')
    parser.add_argument('--no_print_time', dest='print_time', action='store_false', help='Do not print iteration time')
    parser.add_argument('--restart', dest='resume', action='store_false', default=True, help='Restart simulation from t=0, ignoring checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True, help='Resume simulation from latest available checkpoint')
    args = parser.parse_args()
    
    solver = Conforming_solver(moving=args.moving, type_obstacle=args.obstacle, print_iteration_time=args.print_time)
    solver.conforming_solve(dt=args.dt, t_final=args.t_final, resume=args.resume)