import sys
import os
from time import time, perf_counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from math import cos, pi as PI

from user_inputs import *
import user_inputs.user_parameters as user_parameters
from domain_settings import create_bcs_penalty, time_varying_bc
from obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle, BufferObstacle
from post_processing import (
    save_VTK, save_checkpoint, plot_results, create_output_folders,
    find_latest_checkpoint, load_checkpoint_solution, setup_pvd_resume
)
from Solvers.Stokes_solver import solve_stokes_initial


class RIIS_solver:

    def __init__(self, moving=False, type_obstacle="cylinder", n=None, R=None, Re=None, eps=None, structured=True, print_iteration_time=None):
        self.moving = moving
        self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.R = R if R is not None else getattr(user_parameters, 'R', 1000.0)
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.eps = eps if eps is not None else (8.0 / self.n)
        self.structured = structured
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6
        self.print_iteration_time = print_iteration_time if print_iteration_time is not None else getattr(solver_options, 'print_iteration_time', True)

    def RIIS_solve(self, args=None, mesh=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, u_init=None, dt=None, t_final=None, print_iteration_time=None, resume=None):

        if print_iteration_time is not None:
            self.print_iteration_time = print_iteration_time

        if resume is None:
            resume = getattr(solver_options, 'resume_simulation', True)

        # start total timer
        t_start = time()

        eps_val = self.eps if self.eps is not None else (8.0 / self.n)

        # ==================================
        # CREATE MESH & OBSTACLE
        # ==================================
        if obstacle is not None:
            self.obstacle = obstacle
            if hasattr(self.obstacle, 'eps'):
                self.obstacle.eps = eps_val
            if isinstance(obstacle, BufferObstacle):
                self.type_obstacle = "buffer"
            elif isinstance(obstacle, squareObstacle):
                self.type_obstacle = "square"
            elif isinstance(obstacle, circleObstacle):
                self.type_obstacle = "cylinder"
            elif isinstance(obstacle, (lineObstacle, rotatingLineObstacle)):
                self.type_obstacle = "line"
        else:
            if self.type_obstacle not in ["line", "rotating", "rotating_line"]:
                if self.type_obstacle == "cylinder":
                    print("\nObstacle: Cylinder")
                    self.obstacle = circleObstacle(x_obs, y_obs, r_obs, riis_epsilon=eps_val)
                elif self.type_obstacle == "square":
                    print("\nObstacle: Square")
                    self.obstacle = squareObstacle(x_obs, y_obs, side_length, riis_epsilon=eps_val)

                if abs(y_obs - 0.5 * Ly) < 1e-6:
                    print("\nSymmetric configuration: obstacle centered in the channel")
                    self.symmetric = True
                else:
                    print("\nAsymmetric configuration: obstacle moved higher in the channel")
                    self.symmetric = False
            else:
                self.symmetric = False
                if self.type_obstacle == "line":
                    print("\nObstacle: Line")
                    self.obstacle = lineObstacle(xA, yA, xB, yB, riis_epsilon=eps_val, thickness=line_thickness)
                else:
                    print("\nObstacle: Rotating Line")
                    self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, riis_epsilon=eps_val, thickness=line_thickness)

        if mesh is None:
            if self.structured:
                mesh = RectangleMesh(self.n, max(4, int(round(self.n * Ly / Lx))), Lx, Ly)
            else:
                mesh = unstructured_rectangle_mesh(0.0, Lx, 0.0, Ly, self.n, Ly_ref=Ly)

        # ==================================
        # DATA AND SOLVER
        # ==================================
        tol = 1e-10

        T_end = float(t_final) if t_final is not None else 10.0
        dt = float(dt) if dt is not None else 0.1
        num_steps = max(1, int(round(T_end / dt)))
        
        # Reynolds number
        Re = self.Re

        # Density   
        rho = 1.0  

        # Characteristic velocity
        u_char = 1.0              # mean velocity

        # Characteristic length
        L_char = self.obstacle.get_characteristic_length() if hasattr(self.obstacle, 'get_characteristic_length') else 1.0

        # Dynamic viscosity
        mu = rho * L_char * u_char / Re

        print(f"\nCharacteristic length L_char = {L_char}")
        print(f"\nReynolds number Re = {Re} computed with u_characteristic = {u_char}\n")

        f = f_custom(mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0, 0)))
        u_ex_val = u_exact(mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(mesh) if callable(g_custom) else g_custom
        t = Constant(0.0)

        R = self.R
        eps = self.obstacle.eps if hasattr(self.obstacle, 'eps') else eps_val

        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        # Define boundary conditions
        if u_ex_val is not None:
            if self.type_obstacle == "buffer":
                # For buffer obstacle, boundary 1 (x = -L_buf) has homogeneous Neumann (no Dirichlet condition applied)
                bcs = [
                    DirichletBC(W.sub(0), u_ex_val, 3),
                    DirichletBC(W.sub(0), u_ex_val, 4)
                ]
            else:
                bcs = [
                    DirichletBC(W.sub(0), u_ex_val, 1),
                    DirichletBC(W.sub(0), u_ex_val, 3),
                    DirichletBC(W.sub(0), u_ex_val, 4)
                ]
            if g_ex_val is None:
                bcs.append(DirichletBC(W.sub(0), u_ex_val, 2))
                if p_ex_val is not None:
                    bcs.append(DirichletBC(W.sub(1), p_ex_val, 2))
        else:
            bcs = create_bcs_penalty(W, mesh, type_obstacle=self.type_obstacle)

        # Define trial and test functions
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        # Define functions for solutions at previous and current time steps
        uh_n = Function(V)
        sol = Function(W)
        uh, ph = sol.subfunctions

        # Define expressions for RIIS
        phi_expr = self.obstacle.distExpr(mesh, t)
        delta_expr = self.obstacle.delta(mesh, t)
        if u_ex_val is not None:
            us_expr = u_ex_val
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_field'):
            us_expr = self.obstacle.us_field(mesh, t)
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
            us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))
        else:
            us_expr = Constant((0.0, 0.0))


        w = Constant((0.0, 0.0))

        # ==================================
        # DEFINE VARIATIONAL PROBLEM
        # ==================================
        a = Constant(rho)/Constant(dt)*inner(u, v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + 2.0 * Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx \
              + Constant(R / eps) * inner(u, v) * delta_expr * dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx \
              + inner(f, v)*dx \
              + Constant(R / eps) * inner(us_expr, v) * delta_expr * dx

        if g_ex_val is not None:
            ds_b = Measure("ds", domain=mesh)
            L += inner(g_ex_val, v)*ds_b(2)

        # =========================================
        # Create output folders
        # =========================================
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'symmetric': self.symmetric,
            'n': self.n,
            'R': R,
            'Re': Re,
            'is_mms': (u_exact is not None),
        }
        basedir, file_dict = create_output_folders('RIIS', params, extra_fields=['phi', 'delta'])

        DG1 = FunctionSpace(mesh, 'DG', 1)
        phiFun = Function(DG1)
        deltaFun = Function(DG1)

        # ----------------------------------
        # Checkpoint Discovery & Resume
        # ----------------------------------
        resuming = False
        start_step = 0
        latest_t = None

        if resume:
            latest_t, vel_file, press_file, _ = find_latest_checkpoint(basedir)
            if latest_t is not None:
                if latest_t >= T_end - 1e-9:
                    print(f"\n--- RIIS simulation already completed up to t = {latest_t:.2f}s (target T_end = {T_end:.2f}s) in {basedir} ---", flush=True)
                    load_checkpoint_solution(vel_file, press_file, target_u=uh_n, target_p=ph)
                    return mesh, uh_n, ph
                else:
                    print(f"\n--- Resuming RIIS simulation from t = {latest_t:.2f}s up to T_end = {T_end:.2f}s (found checkpoint in {basedir}) ---", flush=True)
                    load_checkpoint_solution(vel_file, press_file, target_u=uh_n, target_p=ph)
                    uh.assign(uh_n)
                    resuming = True
                    setup_pvd_resume(basedir, file_dict, latest_t)

        if resuming:
            t_val = latest_t
            start_step = int(round(latest_t / dt))
            time_varying_bc(latest_t)
            t.assign(latest_t)
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

            phiFun.interpolate(phi_expr)
            deltaFun.interpolate(delta_expr)

            save_VTK(file_dict, t_val, uh, ph, phi=phiFun, delta=deltaFun)
            save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph, phi=phiFun, delta=deltaFun)

        for step in range(start_step, num_steps):
            t_step_start = perf_counter()
            t_val = round((step + 1) * dt, 10)
            print('t =', t_val)
            t.assign(t_val)
            time_varying_bc(t_val)

            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            phiFun.interpolate(phi_expr)
            deltaFun.interpolate(delta_expr)
            
            save_VTK(file_dict, t_val, uh, ph, phi=phiFun, delta=deltaFun)

            # Update previous solution
            uh_n.assign(uh)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max(), flush=True)

            t_step_duration = perf_counter() - t_step_start
            if self.print_iteration_time:
                print(f"\tTempo impiegato per l'iterazione {step + 1}/{num_steps}: {t_step_duration:.4f} s", flush=True)
            
            save_checkpoint(basedir, t_val, mesh=None, moving=self.moving, velocity=uh, pressure=ph, phi=phiFun, delta=deltaFun)
            plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush=True)
        return mesh, uh, ph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes RIIS solver script')
    parser.add_argument('--moving', action='store_true', default=True, help='Use moving obstacle')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'square', 'line', 'rotating', 'rotating_line'],
                        help='Type of obstacle to use in the simulation.')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size (default: 0.1)')
    parser.add_argument('--t_final', type=float, default=10.0, help='Final simulation time (default: 10.0)')
    parser.add_argument('--print_time', action='store_true', default=None, help='Print iteration time')
    parser.add_argument('--restart', dest='resume', action='store_false', default=True, help='Restart simulation from t=0, ignoring checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True, help='Resume simulation from latest available checkpoint')
    args = parser.parse_args()

    # Istanziamo la classe e chiamiamo il solver
    solver = RIIS_solver(moving=args.moving, type_obstacle=args.obstacle, print_iteration_time=args.print_time)
    solver.RIIS_solve(dt=args.dt, t_final=args.t_final, resume=args.resume)
