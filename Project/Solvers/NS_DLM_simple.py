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

    def __init__(self, moving=True, type_obstacle="cylinder", n=None, Re=None):  

        self.moving = moving # Questo verrà sovrascritto per gli ostacoli fissi
        self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6

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


    def NS_DLM_Solve(self, args=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, t_final=None):

        # Start the timer for the simulation
        timer_total.start()
        if args and hasattr(args, "velocity_degree") and args.velocity_degree:
            fem_degree.update({"velocity_degree": args.velocity_degree})
        # Create the meshes
        
        fluid_mesh = create_fluid_mesh(Lx, Ly, self.n)
        # ==================================
        # DATA AND SOLVER
        # ==================================

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

        print("\nCharacteristic length L_char = {}".format(L_char))
        print("\nReynolds number Re = {} computed with u_characteristic = {}\n".format(Re, u_char))

        f = f_custom(fluid_mesh.mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        u_ex_val = u_exact(fluid_mesh.mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(fluid_mesh.mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(fluid_mesh.mesh) if callable(g_custom) else g_custom
        t = Constant(0.0)

        # Create the solid mesh based on the obstacle type
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
        if u_ex_val is not None:
            bcs = [
                DirichletBC(FS['fluid'][0], u_ex_val, 1),
                DirichletBC(FS['fluid'][0], u_ex_val, 3),
                DirichletBC(FS['fluid'][0], u_ex_val, 4)
            ]
            if p_ex_val is not None and g_ex_val is None:
                bcs.append(DirichletBC(FS['fluid'][1], p_ex_val, 2))
            bcs_correction = [
                DirichletBC(V, u_ex_val, 1),
                DirichletBC(V, u_ex_val, 3),
                DirichletBC(V, u_ex_val, 4)
            ]
            if g_ex_val is None:
                bcs.insert(1, DirichletBC(FS['fluid'][0], u_ex_val, 2))
                bcs_correction.insert(1, DirichletBC(V, u_ex_val, 2))
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
            + Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx_fluid \
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
            'symmetric': self.symmetric, 
            'n': self.n,
            'Re': Re,
            'is_mms': (u_exact is not None),
        }
        basedir, file_dict = create_output_folders('DLM', params)

        if u_exact is not None:
            # Steady-state non-linear solve via Newton's method for exact MMS validation
            F_mms = Constant(rho)*inner(dot(u_star, nabla_grad(u_star)), v)*dx_fluid \
                  + Constant(mu)*inner(sym(grad(u_star)), sym(grad(v)))*dx_fluid \
                  - div(v)*ph*dx_fluid + div(u_star)*q*dx_fluid \
                  - inner(f, v)*dx_fluid
            if g_ex_val is not None:
                ds_b = Measure("ds", domain=fluid_mesh.mesh)
                F_mms -= inner(g_ex_val, v)*ds_b(2)

            solve(F_mms == 0, sol_star, bcs=bcs, solver_parameters={
                'snes_type': 'newtonls',
                'snes_rtol': 1e-8,
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            })
            save_VTK(file_dict, T_end, u_star, ph)
            save_checkpoint(basedir, T_end, fluid_mesh.mesh, self.moving, velocity=u_star, pressure=ph)
            timer_total.stop()
            print("Total wall time = {} seconds\n".format(timer_total.elapsed_time), flush=True)
            return

        # ----------------------------------
        # Time-stepping Loop
        # ----------------------------------

        t_val = 0.0
        uh_n.assign(0.0)

        for step in range(num_steps):
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)

            time_varying_bc(t_val)

            """
            !!! Old code for moving and updating the solid mesh coordinates and velocity !!!

            It works only for the cylinder obstacle

            # Update solid position and velocity
            x_solid = SpatialCoordinate(solid_mesh.mesh)
            Dp_old.assign(Dp_new)
            
            # Compute the displacement Dp_[0]
            displ_x = (amplitude * 0.5 * (1.0 - math.cos(0.2 * math.pi * t_val)))
            displ_y = 0.0
            Dp_new.interpolate(as_vector([displ_x, 0.0]))
            # Compute the incremental displacement Dp_[1] = Dp_new - Dp_old
            Dp_inc.assign(Dp_new - Dp_old)

            # Move solid mesh coordinates
            solid_mesh.coordinates.assign(init_coords + Dp_new)
            us_.assign(Dp_inc / Constant(dt))

            """

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
            print('\tu_max:', uh.dat.data.max())
            # ----------------------------------


            # coords = solid_mesh.mesh.coordinates.dat.data_ro
            # print("Solid mesh center:", coords.mean(axis=0), "y-range:", coords[:,1].min(), coords[:,1].max())


            # ------- Create output directories ------- 
            dir_vtk = os.path.join(basedir, "vtk")
            dir_plots_with_solid = os.path.join(basedir, "plots_with_solid")
            dir_plots_fluid_only = os.path.join(basedir, "plots_fluid_only")
            for path in [dir_vtk, dir_plots_with_solid, dir_plots_fluid_only]:
                os.makedirs(path, exist_ok=True)
            
            # ------- Save output & plot (with solid mesh) -------
            save_VTK(file_dict, t_val, uh, ph)
            save_checkpoint(basedir, t_val, mesh=fluid_mesh.mesh, moving=self.moving, velocity=uh, pressure=ph)
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