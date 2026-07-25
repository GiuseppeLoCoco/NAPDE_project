from firedrake import *
from domain_settings import *
from user_inputs import *
from interpolation import *
from time import time, perf_counter
import numpy as np
import math, os, operator, copy, sys, io, json, matplotlib
matplotlib.use('Agg')
from matplotlib import rc, pylab as plt


class Timer:
    def __init__(self):
        self.t0 = 0.0

    def start(self):
        self.t0 = perf_counter()

    def stop(self):
        return perf_counter() - self.t0


timer_total = Timer()


def saveVTK(file_dict, t, uh, ph, phi=None, chi=None):
    uh.rename('u', 'u')
    ph.rename('p', 'p')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    if phi is not None and chi is not None:
        phi.rename('phi', 'phi')
        chi.rename('chi', 'chi')
        file_dict['phi'].write(phi, time=t)
        file_dict['chi'].write(chi, time=t)


class NS_DLM_Solver:

    def __init__(self, conforming=False, moving=True):
        self.conforming = conforming
        self.moving = moving

    def NS_DLM_Solve(self, args=None):
        timer_total.start()
        if args and hasattr(args, "velocity_degree") and args.velocity_degree:
            fem_degree.update({"velocity_degree": args.velocity_degree})

        # Create the mesh
        fluid_mesh = create_fluid_mesh(Lx, Ly, n)
        solid_mesh = create_solid_mesh(x_obs, y_obs, r_obs)

        # ==================================
        # DATA AND SOLVER
        # ==================================

        tol = 1e-10
        T_end = 1.0             # final time
        num_steps = 10          # number of time steps
        dt = T_end / num_steps  # time step size
        mu = 0.1                # dynamic viscosity
        rho = 1.0  

        f = Constant((0.0, 0.0))
        t = Constant(0.0)

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

        # --------------------------------
        # Prescribed kinematics for the solid
        # --------------------------------

        R = VectorFunctionSpace(solid_mesh.mesh, 'P', fem_degree['displacement_degree'])  
        Z = VectorFunctionSpace(solid_mesh.mesh, 'P', fem_degree['lagrange_degree']) 
        Dp_ = [Function(R) for _ in range(3)]
        us_ = Function(R) # solid velocity
        dx_solid = Measure("dx", domain=solid_mesh.mesh)
        ds_solid = Measure("ds", domain=solid_mesh.mesh)

        # Compute Amplitude
        coords = solid_mesh.mesh.coordinates.dat.data_ro
        diameter = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
        amplitude = 6.0 * diameter

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
        bcs = create_boundary_conditions(fluid_mesh, **FS)

        # ---------------------------------
        # Delta-interpolation for Fluid-Structure interaction (Firedrake)
        # ---------------------------------
        fsi_interpolation = FSIInterpolation()
        fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
        fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

        # ---------------------------------
        # DEFINE VARIATIONAL PROBLEMS
        # ---------------------------------

        # Step 1: tentative velocity (DLM-NS-S1)
        a1 = Constant(rho)/Constant(dt)*inner(u, v)*dx_fluid \
            + Constant(rho)*inner(dot(uh_n, nabla_grad(u)), v)*dx_fluid \
            + 0.5*Constant(rho)*div(uh_n)*inner(u, v)*dx_fluid \
            + 2.0*Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx_fluid \
            - div(v)*p*dx_fluid \
            + div(u)*q*dx_fluid

        L1 = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx_fluid \
            + inner(f, v)*dx_fluid \
            - inner(Lm_f, v)*dx_fluid

        # Step 2: Lagrange multiplier (DLM-NS-S2)
        a2 = inner(Lm, e) * dx_solid
        L2 = (1.0 / Constant(dt)) * inner(uf_ - us_, e) * dx_solid \
              + inner(Lm_[1], e) * dx_solid

        # Step 3: Velocity correction (DLM-NS-S3)
        u_v = TrialFunction(V)
        v_v = TestFunction(V)
        uh = Function(V)

        a3 = Constant(rho)/Constant(dt)*inner(u_v, v_v)*dx_fluid
        L3 = Constant(rho)/Constant(dt)*inner(u_star, v_v)*dx_fluid \
              + inner(Lm_f - Lm_f_old, v_v)*dx_fluid

        # Time-stepping
        t_val = 0.0
        uh_n.assign(0.0)

        for step in range(num_steps):
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)

            time_varying_bc(t_val)

            # Update Lagrange multiplier for new time step
            Lm_[1].assign(Lm_[0])

            # Interpolate Lagrange multiplier from solid mesh to fluid mesh
            Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            # STEP 1: Solve tentative velocity
            solve(a1 == L1, sol_star, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            # Interpolate velocity onto solid mesh
            uf_.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, u_star, "S"))

            # Update solid position and velocity
            x_solid = SpatialCoordinate(solid_mesh.mesh)
            Dp_[2].assign(Dp_[1])

            # Calcola il nuovo spostamento Dp_[0]
            displ_x = (amplitude * 0.5 * (1.0 - math.cos(0.2 * math.pi * t_val)))
            displ_y = 0.0
            Dp_[0].interpolate(as_vector([displ_x, displ_y]) + 0*x_solid)

            # Calcola lo spostamento incrementale Dp_[1] = Dp_new - Dp_old
            Dp_[1].assign(Dp_[0] - Dp_[2])

            # Move solid mesh coordinates
            solid_mesh.mesh.coordinates.assign(solid_mesh.mesh.coordinates + Dp_[1])

            us_.assign(Dp_[1] / Constant(dt))

            # STEP 2: Solve Lagrange multiplier
            solve(a2 == L2, Lm_[0], solver_parameters={'ksp_type': 'bcgs', 'pc_type': 'sor'})

            # Interpolate Lagrange multiplier
            Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[0], "F"))
            Lm_f_old.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

            # STEP 3: Solve velocity correction
            solve(a3 == L3, uh, solver_parameters={'ksp_type': 'cg', 'pc_type': 'sor'})
            uh_n.assign(uh)

        wall_time = timer_total.stop()
        print("Total simulation wall time : {} sec".format(wall_time), "\n", flush=True)


if __name__ == "__main__":
    solver = NS_DLM_Solver()
    solver.NS_DLM_Solve()