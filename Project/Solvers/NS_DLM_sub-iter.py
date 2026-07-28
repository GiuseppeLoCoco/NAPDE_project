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
import math
import numpy as np
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders
from domain_settings.obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle

class Timer:
    def __init__(self):
        self.t0 = 0.0

    def start(self):
        self.t0 = perf_counter()

    def stop(self):
        return perf_counter() - self.t0


timer_total = Timer()

class NS_DLM_Solver:

    def __init__(self, moving=True, type_obstacle="cylinder"):  

        self.moving = moving # Questo verrà sovrascritto per gli ostacoli fissi
        self.unsteady = True
        self.instationary = True
        self.mean = True
        self.type_obstacle = type_obstacle

        # Inizializza l'ostacolo e imposta le proprietà in base al suo tipo
        if self.type_obstacle == "cylinder":
            print("\nObstacolo: Cilindro")
            self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
            if x_obs == y_obs:
                print("Configurazione simmetrica: cilindro centrato nel canale")
                self.symmetric = True
            else:
                print("Configurazione asimmetrica: cilindro spostato più in alto nel canale")
                self.symmetric = False
        elif self.type_obstacle == "square":
            print("\nObstacolo: Quadrato")
            self.obstacle = squareObstacle(x_obs, y_obs, side_length)
            self.moving = False  # L'ostacolo quadrato è tipicamente fisso nel contesto DLM
            if x_obs == y_obs: # Assumendo che il centro del quadrato sia x_obs, y_obs
                print("Configurazione simmetrica: quadrato centrato nel canale")
                self.symmetric = True
            else:
                print("Configurazione asimmetrica: quadrato spostato più in alto nel canale")
                self.symmetric = False
        elif self.type_obstacle == "line":
            print("\nObstacolo: Linea")
            # Ordine corretto degli argomenti per lineObstacle
            self.obstacle = lineObstacle(xA, yA, xB, yB, riis_epsilon=line_thickness, thickness=line_thickness)
            self.moving = False  # Ostacolo linea fisso
            self.symmetric = False # Gli ostacoli linea non sono generalmente simmetrici in questo contesto
        elif self.type_obstacle == "rotating_line":
            print("\nObstacolo: Linea Rotante")
            # Ordine corretto degli argomenti per rotatingLineObstacle
            self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, riis_epsilon=line_thickness, thickness=line_thickness)
            self.symmetric = False # Gli ostacoli linea rotanti non sono generalmente simmetrici
        else:
            raise ValueError(f"Tipo di ostacolo non supportato: {self.type_obstacle}")


    def NS_DLM_Solve(self, args=None):

        # Start the timer for the simulation
        timer_total.start()
        if args and hasattr(args, "velocity_degree") and args.velocity_degree:
            fem_degree.update({"velocity_degree": args.velocity_degree})
        # Create the meshes
        fluid_mesh = create_fluid_mesh(Lx, Ly, n)
        # ==================================
        # DATA AND SOLVER
        # ==================================

        tol = 1e-10
        T_end = 20.0             # final time
        num_steps = 40           # number of time steps
        dt = T_end / num_steps   # time step size

        # Reynolds number
        if self.unsteady:
            Re = 80
        else:
            Re = 40

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

        if self.unsteady == True:
            print("\nReynolds number Re = {} --> Unsteady Regime\n", format(Re))
        else:
            print("\nReynolds number Re = {} --> Steady Regime\n", format(Re))

        f = Constant((0.0, 0.0))
        t = Constant(0.0)

        # Create the solid mesh based on the obstacle type
        solid_mesh = create_solid_mesh(self.obstacle, n)

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
        Dp_inc = Function(R)
        us_ = Function(R) # solid velocity
        dx_solid = Measure("dx", domain=solid_mesh)
        ds_solid = Measure("ds", domain=solid_mesh)

        # Compute Amplitude
        coords = solid_mesh.coordinates.dat.data_ro
        bbox = coords.max(axis=0) - coords.min(axis=0)
        diameter = bbox.max()
        amplitude = 6.0 * diameter

        # Initialize solid mesh coordinates
        init_coords = Function(R).interpolate(solid_mesh.coordinates)


        # --------------------------------
        # Initialize Lagrange Multiplier Variational Problem
        # --------------------------------

        Lm = TrialFunction(Z)
        e = TestFunction(Z)
        uf_ = Function(R)  # fluid velocity interpolated on solid mesh  
        us_.assign(0.0)

        Lm_ = [Function(Z), Function(Z)]
        Lm_[0].assign(0.0)
        Lm_[1].assign(0.0)

        # Create boundary conditions dictionary and setup
        FS = {'fluid': [W.sub(0), W.sub(1), Z1], 'lagrange': [Z]}
        bcs = create_boundary_conditions(fluid_mesh, type_obstacle=self.type_obstacle, **FS)

        bcs_correction = create_boundary_conditions_correction(fluid_mesh, V, type_obstacle=self.type_obstacle)

        # ---------------------------------
        # Delta-interpolation for Fluid-Structure interaction (Firedrake)
        # ---------------------------------
        fsi_interpolation = FSIInterpolation()
        # fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
        # fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

        # ---------------------------------
        # DEFINE VARIATIONAL PROBLEMS
        # ---------------------------------

        # ------- Step 1: tentative velocity (DLM-NS-S1) -------
        a1 = Constant(rho)/Constant(dt)*inner(u, v)*dx_fluid \
            + Constant(rho)*inner(dot(uh_n, nabla_grad(u)), v)*dx_fluid \
            + 0.5*Constant(rho)*div(uh_n)*inner(u, v)*dx_fluid \
            + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx_fluid \
            - div(v)*p*dx_fluid \
            + div(u)*q*dx_fluid

        L1 = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx_fluid \
            + inner(f, v)*dx_fluid \
            - inner(Lm_f, v)*dx_fluid
        

        # ------- Step 2: Lagrange multiplier (DLM-NS-S2) -------
        a2 = inner(Lm, e) * dx_solid
        L2 = (1.0 / Constant(dt)) * inner(uf_ - us_, e) * dx_solid \
              + inner(Lm_[1], e) * dx_solid


        # ------- Step 3: Velocity correction (DLM-NS-S3) -------
        u_v = TrialFunction(V)
        v_v = TestFunction(V)
        uh = Function(V)

        a3 = Constant(rho)/Constant(dt)*inner(u_v, v_v)*dx_fluid
        L3 = Constant(rho)/Constant(dt)*inner(u_star, v_v)*dx_fluid \
              + inner(Lm_f - Lm_f_old, v_v)*dx_fluid


        # ------- Setup output folders -------
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'unsteady': self.unsteady,
            'symmetric': self.symmetric, 
            'n': n,
            'Re': Re,
        }
        basedir, file_dict = create_output_folders('DLM', params)


        # ----------------------------------
        # Time-stepping Loop
        # ----------------------------------

        t_val = 0.0
        uh_n.assign(0.0)

        for step in range(num_steps):
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)

            # Parameters for DLM Convergence Loop
            sub_tol = 1e-5         # Tollerance on variation of the Lagrange multiplier
            max_sub_iters = 20     # Max number of iterations for sub-iteration loop

            time_varying_bc(t_val)

            # Update solid position and velocity
            Dp_old.assign(Dp_new)

            if self.moving:
                # Update the displacement based on the prescribed kinematics of the obstacle
                dx_expr = self.obstacle.displ_x(t_val)
                dy_expr = self.obstacle.displ_y(t_val)
                
                Dp_new.interpolate(as_vector([dx_expr, dy_expr]))
                
                # Update mesh coorinates based on the new displacement
                solid_mesh.coordinates.assign(init_coords + Dp_new)
                
                # Update solid velocity us_
                us_x_expr = self.obstacle.us_x(t_val)
                us_y_expr = self.obstacle.us_y(t_val)
                us_.interpolate(as_vector([us_x_expr, us_y_expr]))
            else:
                # Fixed obstacle
                Dp_new.assign(0.0)
                us_.assign(0.0)
            

            fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
            fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

            # ============================
            # SUB-ITERATIONS LOOP 
            # ============================
            
            # Initialization for the sub-iteration loop
            sub_iter = 0
            converged = False
            
            # Alla prima sotto-iterazione partiamo dalla soluzione del passo precedente
            # uh_current viene usata come guess per la velocità tentative
            uh_current = Function(V)
            uh_current.assign(uh_n) 

            # Update Lagrange multiplier for new time step
            Lm_[1].assign(Lm_[0])
            # Interpolate Lagrange multiplier from solid mesh to fluid mesh
            Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            # Lm_f_start_step = Function(Z1)
            # Lm_f_start_step.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            while not converged and sub_iter < max_sub_iters:

                sub_iter += 1

                Lm_f_old.assign(Lm_f)

                # ------- STEP 1: Solve tentative velocity -------
                solve(a1 == L1, sol_star, bcs=bcs, 
                        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

                # Interpolate tentative velocity onto solid mesh
                uf_.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, u_star, "S"))

                # ------- STEP 2: Solve Lagrange multiplier -------
                solve(a2 == L2, Lm_[0], solver_parameters={'ksp_type': 'bcgs', 'pc_type': 'sor'})

                # Interpolate Lagrange multiplier onto fluid mesh
                Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[0], "F"))

                # ------- STEP 3: Solve velocity correction -------
                solve(a3 == L3, uh, bcs=bcs_correction, solver_parameters={'ksp_type': 'cg', 'pc_type': 'sor'})

                # ------- Convergence Criterion -------
                # Compute the L2-norm on the increment || Lm_f^(k+1) - Lm_f^(k) ||
                diff_Lm = norm(Lm_f - Lm_f_old)
                norm_Lm = norm(Lm_f) + 1e-12           # To avoid zero division
                rel_increment = diff_Lm / norm_Lm

                print(f"\t [Sub-iteration: {sub_iter}] Relative L_m Increment: {rel_increment:.4e} (Abs: {diff_Lm:.4e})")

                if rel_increment < sub_tol:
                    converged = True
                    print(f"\t Convergece achieved in {sub_iter} iterations.")

                # Update estimated Lagrange Multiplier for the next sub-iteration
                Lm_[1].assign(Lm_[0])

            if not converged:
                print(f"\t WARNING: Sub-iteration not converged within {max_sub_iters} iterations!")
            
            # Update previous solution
            uh_n.assign(uh)

            # ------- Print max velocity -------
            print('\tu_max:', uh.dat.data.max())
            # ----------------------------------

            # ------- Create output directories ------- 
            dir_vtk = os.path.join(basedir, "vtk")
            dir_plots_with_solid = os.path.join(basedir, "plots_with_solid")
            dir_plots_fluid_only = os.path.join(basedir, "plots_fluid_only")
            for path in [dir_vtk, dir_plots_with_solid, dir_plots_fluid_only]:
                os.makedirs(path, exist_ok=True)
            
            # ------- Save output & plot (with solid mesh) -------
            save_VTK(file_dict, t_val, uh, ph)
            save_checkpoint(dir_vtk, t_val, mesh=fluid_mesh.mesh, moving=self.moving, velocity=uh, pressure=ph)
            plot_results(fluid_mesh.mesh, uh, ph, t_val, basedir = dir_plots_with_solid, solid_mesh=solid_mesh)

            # ------- Plot (without solid mesh) -------
            plot_results(fluid_mesh.mesh, uh, ph, t_val=t_val, basedir=dir_plots_fluid_only)

        wall_time = timer_total.stop()
        print("Total simulation wall time : {} sec".format(wall_time), "\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Navier-Stokes DLM Solver')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'square', 'line', 'rotating_line'],
                        help='Type of obstacle to use in the simulation.')
    
    args = parser.parse_args()

    # Istanziamo la classe passando il tipo di ostacolo letto da riga di comando
    solver = NS_DLM_Solver(type_obstacle=args.obstacle)
    solver.NS_DLM_Solve(args)