from curses.ascii import FS

from firedrake import *
import sys
import os
from time import time, perf_counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))
from domain_settings import *
from user_inputs import *
import user_inputs.user_parameters as user_parameters
import math
import numpy as np
from post_processing import (
    save_VTK, save_checkpoint, plot_results, create_output_folders,
    find_latest_checkpoint, load_checkpoint_solution, setup_pvd_resume, _safe_load_mesh
)
from domain_settings.obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle, BufferObstacle
from Solvers.Stokes_solver import solve_stokes_initial

class Timer:
    def __init__(self):
        self.t0 = 0.0

    def start(self):
        self.t0 = perf_counter()

    def stop(self):
        return perf_counter() - self.t0


timer_total = Timer()

class NS_DLM_Solver:

    def __init__(self, moving=True, type_obstacle="cylinder", n=None, Re=None, structured=True, print_iteration_time=None):  

        self.moving = moving # This will be overwritten for fixed obstacles
        self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.structured = structured
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6
        self.print_iteration_time = print_iteration_time if print_iteration_time is not None else getattr(solver_options, 'print_iteration_time', True)

        # Initialize the obstacle based on the type chosen
        if self.type_obstacle == "cylinder":
            print("\nObstacle: Cylinder")
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
            if  self.symmetric:
                print("Symmetric Configuration: cylinder centered in the channel")
                
            else:
                print("Asymmetric Configuration: cylinder shifted upward in the channel")
                
        elif self.type_obstacle == "square":
            print("\nObstacle: Square")
            self.obstacle = squareObstacle(x_obs, y_obs, side_length)
            self.moving = False  # The square obstacle is typically fixed in the DLM context
            if self.symmetric: # Assuming the center of the square is at x_obs, y_obs
                print("Symmetric Configuration: square centered in the channel")
            else:
                print("Asymmetric Configuration: square shifted upward in the channel")
                
        elif self.type_obstacle == "line":
            print("\nObstacle: Line")
            # Correct order of arguments for lineObstacle
            self.obstacle = lineObstacle(xA, yA, xB, yB, riis_epsilon=line_thickness, thickness=line_thickness)
            self.moving = False  # Fixed line obstacle this context
            self.symmetric = False # Line obstacles are generally not symmetric in
        elif self.type_obstacle in ["rotating_line", "rotating"]:
            print("\nObstacle: Rotating Line")
            # Correct order of arguments for rotatingLineObstacle
            self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, riis_epsilon=line_thickness, thickness=line_thickness)
            self.symmetric = False # Line obstacles are generally not symmetric in this context
        elif self.type_obstacle in ["buffer", "Buffer", "none", "None", None]:
            self.obstacle = None
            self.moving = False
            self.symmetric = False
        else:
            raise ValueError(f"Type of obstacle not supported: {self.type_obstacle}")


    def NS_DLM_Solve(self, args=None, fluid_mesh=None, solid_mesh=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, u_init=None, dt=None, t_final=None, print_iteration_time=None, resume=None):

        if print_iteration_time is not None:
            self.print_iteration_time = print_iteration_time

        if resume is None:
            resume = getattr(solver_options, 'resume_simulation', True)

        # Start the timer for the simulation
        timer_total.start()
        if args and hasattr(args, "velocity_degree") and args.velocity_degree:
            fem_degree.update({"velocity_degree": args.velocity_degree})
        
        if obstacle is not None:
            self.obstacle = obstacle
            if isinstance(obstacle, BufferObstacle):
                self.type_obstacle = "buffer"
            elif isinstance(obstacle, squareObstacle):
                self.type_obstacle = "square"
            elif isinstance(obstacle, circleObstacle):
                self.type_obstacle = "cylinder"
            elif isinstance(obstacle, (lineObstacle, rotatingLineObstacle)):
                self.type_obstacle = "line"

        # Create the meshes
        if fluid_mesh is None:
            fluid_mesh = create_fluid_mesh(Lx, Ly, self.n, structured=self.structured)
        elif not hasattr(fluid_mesh, 'mesh'):
            class FluidMeshWrapper:
                def __init__(self, m):
                    self.mesh = m
            fluid_mesh = FluidMeshWrapper(fluid_mesh)
        # ==================================
        # DATA AND SOLVER
        # ==================================

        tol = 1e-10
        T_end = float(t_final) if t_final is not None else 20.0
        dt = float(dt) if dt is not None else 0.1
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

        print("\nCharacteristic length L_char = {}".format(L_char))
        print("\nReynolds number Re = {} computed with u_characteristic = {}\n".format(Re, u_char))

        f = f_custom(fluid_mesh.mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        u_ex_val = u_exact(fluid_mesh.mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(fluid_mesh.mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(fluid_mesh.mesh) if callable(g_custom) else g_custom
        t = Constant(0.0)

        # Create the solid mesh based on the obstacle type
        if solid_mesh is None:
            solid_mesh = create_solid_mesh(self.obstacle, self.n)

        # --------------------------------
        # Initialize Flow Variational Problem
        # --------------------------------

        # Define function spaces
        V = VectorFunctionSpace(fluid_mesh.mesh, 'P', fem_degree['velocity_degree'])
        Q = FunctionSpace(fluid_mesh.mesh, 'P', fem_degree['pressure_degree'])
        Z1 = VectorFunctionSpace(fluid_mesh.mesh, 'P', fem_degree['lagrange_degree'])
        W = V * Q

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        # Define functions for solutions at previous and current time steps
        uh_n = Function(V)
        sol_star = Function(W)
        u_star, ph = sol_star.subfunctions

        Lm_f = Function(Z1)
        Lm_f.assign(0.0)
        Lm_f_old = Function(Z1)

        dx_fluid = Measure("dx", domain=fluid_mesh.mesh)


        # Vorticity for the unsteady case
        vort, psi = Function(Q), Function(Q)
        vort.assign(0.0)
        psi.assign(0.0)

        # --------------------------------
        # Prescribed kinematics for the solid
        # --------------------------------

        R = VectorFunctionSpace(solid_mesh, 'P', fem_degree['displacement_degree'])
        Z = VectorFunctionSpace(solid_mesh, 'P', fem_degree['lagrange_degree'])
        Dp_new = Function(R)
        Dp_old = Function(R)
        us_ = Function(R) # solid velocity
        dx_solid = Measure("dx", domain=solid_mesh)

        # Initialize solid mesh coordinates
        init_coords = Function(R).interpolate(solid_mesh.coordinates)



        # --------------------------------
        # Initialize Lagrange Multiplier Variational Problem
        # --------------------------------

        Lm = TrialFunction(Z)
        e = TestFunction(Z)
        uf_ = Function(R)  # fluid velocity interpolated on solid mesh  
        if u_ex_val is not None:
            us_.interpolate(u_ex_val)
        elif self.obstacle is not None and hasattr(self.obstacle, 'velocity'):
            us_.interpolate(self.obstacle.velocity(init_coords, 0.0))
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
            us_.interpolate(as_vector([self.obstacle.us_x(0.0), self.obstacle.us_y(0.0)]))
        else:
            us_.assign(0.0)

        Lm_ = [Function(Z), Function(Z)]
        Lm_[0].assign(0.0)
        Lm_[1].assign(0.0)

        # Create boundary conditions dictionary and setup
        FS = {'fluid': [W.sub(0), W.sub(1), Z1], 'lagrange': [Z]}
        if u_ex_val is not None:
            if self.type_obstacle == "buffer":
                # For buffer obstacle, boundary 1 (x = -L_buf) has homogeneous Neumann (no Dirichlet condition applied)
                bcs = [
                    DirichletBC(FS['fluid'][0], u_ex_val, 3),
                    DirichletBC(FS['fluid'][0], u_ex_val, 4)
                ]
                bcs_correction = [
                    DirichletBC(V, u_ex_val, 3),
                    DirichletBC(V, u_ex_val, 4)
                ]
            else:
                bcs = [
                    DirichletBC(FS['fluid'][0], u_ex_val, 1),
                    DirichletBC(FS['fluid'][0], u_ex_val, 3),
                    DirichletBC(FS['fluid'][0], u_ex_val, 4)
                ]
                bcs_correction = [
                    DirichletBC(V, u_ex_val, 1),
                    DirichletBC(V, u_ex_val, 3),
                    DirichletBC(V, u_ex_val, 4)
                ]
            if g_ex_val is None:
                bcs.append(DirichletBC(FS['fluid'][0], u_ex_val, 2))
                if p_ex_val is not None:
                    bcs.append(DirichletBC(FS['fluid'][1], p_ex_val, 2))
                bcs_correction.append(DirichletBC(V, u_ex_val, 2))
        else:
            bcs = create_boundary_conditions(fluid_mesh, type_obstacle=self.type_obstacle, **FS)
            bcs_correction = create_boundary_conditions_correction(fluid_mesh, V, type_obstacle=self.type_obstacle)

        # ---------------------------------
        # Delta-interpolation for Fluid-Structure interaction (Firedrake)
        # ---------------------------------
        fsi_interpolation = FSIInterpolation()
        fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
        fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

        # ---------------------------------
        # DEFINE VARIATIONAL PROBLEMS
        # ---------------------------------

        # ------- Step 1: tentative velocity (DLM-NS-S1) -------
        a1 = Constant(rho)/Constant(dt)*inner(u, v)*dx_fluid \
            + Constant(rho)*inner(dot(uh_n, nabla_grad(u)), v)*dx_fluid \
            + 0.5*Constant(rho)*div(uh_n)*inner(u, v)*dx_fluid \
            + 2.0 * Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx_fluid \
            - div(v)*p*dx_fluid \
            + div(u)*q*dx_fluid

        L1 = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx_fluid \
            + inner(f, v)*dx_fluid \
            - inner(Lm_f, v)*dx_fluid

        if g_ex_val is not None:
            ds_b = Measure("ds", domain=fluid_mesh.mesh)
            L1 += inner(g_ex_val, v)*ds_b(2)
        

        # ------- Step 2: Lagrange multiplier (DLM-NS-S2) -------
        a2 = inner(Lm, e) * dx_solid
        L2 = (Constant(rho) / Constant(dt)) * inner(uf_ - us_, e) * dx_solid \
              + inner(Lm_[1], e) * dx_solid

        # ------- Step 3: Velocity correction (DLM-NS-S3) -------
        u_v = TrialFunction(V)
        v_v = TestFunction(V)
        uh = Function(V)

        a3 = Constant(rho)/Constant(dt)*inner(u_v, v_v)*dx_fluid
        
        L3 = Constant(rho)/Constant(dt)*inner(u_star, v_v)*dx_fluid \
              - inner(Lm_f - Lm_f_old, v_v)*dx_fluid


        # ------- Setup output folders -------
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'symmetric': self.symmetric, 
            'n': self.n,
            'Re': Re,
            'is_mms': (u_exact is not None),
        }
        basedir, file_dict = create_output_folders('DLM', params)

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
                    print(f"\n--- DLM simulation already completed up to t = {latest_t:.2f}s (target T_end = {T_end:.2f}s) in {basedir} ---", flush=True)
                    load_checkpoint_solution(vel_file, press_file, target_u=uh_n, target_p=ph)
                    return fluid_mesh.mesh, uh_n, ph
                else:
                    print(f"\n--- Resuming DLM simulation from t = {latest_t:.2f}s up to T_end = {T_end:.2f}s (found checkpoint in {basedir}) ---", flush=True)
                    load_checkpoint_solution(vel_file, press_file, target_u=uh_n, target_p=ph)
                    uh.assign(uh_n)
                    resuming = True
                    setup_pvd_resume(basedir, file_dict, latest_t)

                    # Restore Lagrange multiplier if saved in checkpoint
                    lagrange_file = os.path.join(basedir, 'lagrange', f'lagrange_t={latest_t:.2f}.h5')
                    if os.path.exists(lagrange_file):
                        try:
                            with CheckpointFile(lagrange_file, 'r') as chk_l:
                                mesh_l = _safe_load_mesh(chk_l)
                                lm_chk = chk_l.load_function(mesh_l, name="lagrange")
                            Lm_[0].dat.data[:] = lm_chk.dat.data_ro[:]
                        except Exception as e:
                            print(f"Warning: could not load Lagrange multiplier checkpoint ({e}), initializing to 0.", flush=True)
                            Lm_[0].assign(0.0)

                    # Restore solid position and velocity at latest_t
                    if self.moving and self.obstacle is not None:
                        if hasattr(self.obstacle, 'displacement'):
                            Dp_new.interpolate(self.obstacle.displacement(init_coords, latest_t))
                        else:
                            dx_expr = self.obstacle.displ_x(latest_t)
                            dy_expr = self.obstacle.displ_y(latest_t)
                            Dp_new.interpolate(as_vector([dx_expr, dy_expr]))
                        solid_mesh.coordinates.assign(init_coords + Dp_new)

                        if hasattr(self.obstacle, 'velocity'):
                            us_.interpolate(self.obstacle.velocity(init_coords, latest_t))
                        elif hasattr(self.obstacle, 'us_x'):
                            us_x_expr = self.obstacle.us_x(latest_t)
                            us_y_expr = self.obstacle.us_y(latest_t)
                            us_.interpolate(as_vector([us_x_expr, us_y_expr]))

        # ----------------------------------
        # Time-stepping Loop Setup
        # ----------------------------------
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
                    uh_n.interpolate(u_init(fluid_mesh.mesh))
                else:
                    uh_n.assign(u_init)
            else:
                print("Initializing velocity with stationary Stokes solver (t=0)...", flush=True)
                uh_stokes, _ = solve_stokes_initial(
                    mesh=fluid_mesh.mesh, bcs=bcs, mu=mu, f_custom=f_custom, g_custom=g_custom, W=W
                )
                uh_n.assign(uh_stokes)
            uh.assign(uh_n)
            save_VTK(file_dict, 0.0, uh_n, ph)
            save_checkpoint(basedir, 0.0, mesh=fluid_mesh.mesh, moving=False, velocity=uh_n, pressure=ph, lagrange=Lm_[0])

        for step in range(start_step, num_steps):
            t_step_start = perf_counter()
            t_val = round((step + 1) * dt, 10)
            print('t =', t_val)
            t.assign(t_val)

            time_varying_bc(t_val)

            # Update solid position and velocity
            Dp_old.assign(Dp_new)

            if self.moving and self.obstacle is not None:
                # Update displacement and solid mesh coordinates
                if hasattr(self.obstacle, 'displacement'):
                    Dp_new.interpolate(self.obstacle.displacement(init_coords, t_val))
                else:
                    dx_expr = self.obstacle.displ_x(t_val)
                    dy_expr = self.obstacle.displ_y(t_val)
                    Dp_new.interpolate(as_vector([dx_expr, dy_expr]))

                solid_mesh.coordinates.assign(init_coords + Dp_new)

                # Update solid velocity us_
                if hasattr(self.obstacle, 'velocity'):
                    us_.interpolate(self.obstacle.velocity(init_coords, t_val))
                elif hasattr(self.obstacle, 'us_x'):
                    us_x_expr = self.obstacle.us_x(t_val)
                    us_y_expr = self.obstacle.us_y(t_val)
                    us_.interpolate(as_vector([us_x_expr, us_y_expr]))
            else:
                # Fixed obstacle
                Dp_new.assign(0.0)
                if u_ex_val is not None:
                    us_.interpolate(u_ex_val)
                elif self.obstacle is not None and hasattr(self.obstacle, 'velocity'):
                    us_.interpolate(self.obstacle.velocity(init_coords, t_val))
                elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
                    us_.interpolate(as_vector([self.obstacle.us_x(t_val), self.obstacle.us_y(t_val)]))
                else:
                    us_.assign(0.0)


            

            # Update Lagrange multiplier for new time step
            Lm_[1].assign(Lm_[0])
            # Interpolate Lagrange multiplier from solid mesh to fluid mesh
            Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            # ------- STEP 1: Solve tentative velocity -------
            solve(a1 == L1, sol_star, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Interpolate velocity onto solid mesh
            uf_.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, u_star, "S"))

            # ------- STEP 2: Solve Lagrange multiplier -------
            solve(a2 == L2, Lm_[0], solver_parameters={'ksp_type': 'bcgs', 'pc_type': 'sor'})

            # Interpolate Lagrange multiplier
            Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[0], "F"))
            Lm_f_old.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            # ------- STEP 3: Solve velocity correction -------
            solve(a3 == L3, uh, bcs=bcs_correction, solver_parameters={'ksp_type': 'cg', 'pc_type': 'sor'})
            # Update previous solution
            uh_n.assign(uh)

            # ------- Print max velocity -------
            print('\tu_max:', uh.dat.data.max(), flush=True)
            # ----------------------------------

            t_step_duration = perf_counter() - t_step_start
            if self.print_iteration_time:
                print(f"\tTempo impiegato per l'iterazione {step + 1}/{num_steps}: {t_step_duration:.4f} s", flush=True)

            # Save solution to file (VTK/PVD), checkpoint, and plots
            save_VTK(file_dict, t_val, uh, ph)
            save_checkpoint(basedir, t_val, mesh=fluid_mesh.mesh, moving=False, velocity=uh, pressure=ph, lagrange=Lm_[0])
            plot_results(fluid_mesh.mesh, uh, ph, t_val=t_val, basedir=basedir, solid_mesh=solid_mesh)


        wall_time = timer_total.stop()
        print("Total simulation wall time : {} sec".format(wall_time), "\n", flush=True)
        return fluid_mesh.mesh, uh, ph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Navier-Stokes DLM Solver')
    parser.add_argument('--moving', action='store_true', default=True, help='Use moving obstacle')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'square', 'line', 'rotating', 'rotating_line'],
                        help='Type of obstacle to use in the simulation.')
    
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size (default: 0.1)')
    parser.add_argument('--t_final', type=float, default=20.0, help='Final simulation time (default: 20.0)')
    parser.add_argument('--print_time', action='store_true', default=None, help='Print iteration time')
    parser.add_argument('--no_print_time', dest='print_time', action='store_false', help='Do not print iteration time')
    
    parser.add_argument('--restart', dest='resume', action='store_false', default=True, help='Restart simulation from t=0, ignoring checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True, help='Resume simulation from latest available checkpoint')
    
    args = parser.parse_args()
    solver = NS_DLM_Solver(moving=args.moving, type_obstacle=args.obstacle, print_iteration_time=args.print_time)
    solver.NS_DLM_Solve(dt=args.dt, t_final=args.t_final, resume=args.resume)