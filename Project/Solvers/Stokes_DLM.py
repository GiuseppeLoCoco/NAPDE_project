import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *

from user_inputs import *
import user_inputs.user_parameters as user_parameters
from domain_settings import (
    create_boundary_conditions,
    create_boundary_conditions_correction,
    FSIInterpolation,
    interpolate_nonmatching_mesh_delta
)
from domain_settings.mesh_settings import create_fluid_mesh, create_solid_mesh
from obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle

class Stokes_DLM_Solver:

    def __init__(self, type_obstacle="square", n=None, Re=None):
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)

    def Stokes_DLM_Solve(self, fluid_mesh=None, solid_mesh=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None):
        """
        Solves the stationary Stokes problem using the 3-step DLM operator splitting
        (Fluid-Structure Interaction with non-matching meshes) without time-stepping loop.
        Returns (uh, ph) as initial condition for velocity/pressure.
        """
        # ==================================
        # CREATE OBSTACLE & MESHES (if not provided)
        # ==================================
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            if self.type_obstacle in ["circle", "cylinder"]:
                self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
            elif self.type_obstacle == "square":
                self.obstacle = squareObstacle(x_obs, y_obs, side_length)
            elif self.type_obstacle == "line":
                self.obstacle = lineObstacle(xA, xB, yA, yB)
            elif self.type_obstacle == "rotating":
                self.obstacle = rotatingLineObstacle(xA, xB, yA, yB)
            else:
                raise ValueError(f"Unsupported obstacle type: {self.type_obstacle}")

        if fluid_mesh is None:
            fluid_mesh = create_fluid_mesh(Lx, Ly, self.n)

        if solid_mesh is None:
            solid_mesh = create_solid_mesh(self.obstacle, self.n)

        # ==================================
        # PARAMETERS & DATA
        # ==================================
        rho = 1.0
        u_char = 1.0
        L_char = self.obstacle.get_characteristic_length()
        mu = rho * L_char * u_char / self.Re
        t = Constant(0.0)
 
        f = f_custom(fluid_mesh.mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        u_ex_val = u_exact(fluid_mesh.mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(fluid_mesh.mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(fluid_mesh.mesh) if callable(g_custom) else g_custom

        # ==================================
        # FUNCTION SPACES
        # ==================================
        V = VectorFunctionSpace(fluid_mesh.mesh, 'P', fem_degree['velocity_degree'])
        Q = FunctionSpace(fluid_mesh.mesh, 'P', fem_degree['pressure_degree'])
        Z1 = VectorFunctionSpace(fluid_mesh.mesh, 'P', fem_degree['lagrange_degree'])
        W = V * Q

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        sol_star = Function(W)
        u_star, ph = sol_star.subfunctions

        Lm_f = Function(Z1)
        Lm_f.assign(0.0)
        Lm_f_old = Function(Z1)
        Lm_f_old.assign(0.0)

        dx_fluid = Measure("dx", domain=fluid_mesh.mesh)

        # Solid function spaces
        R = VectorFunctionSpace(solid_mesh, 'P', fem_degree['displacement_degree'])
        Z = VectorFunctionSpace(solid_mesh, 'P', fem_degree['lagrange_degree'])
        us_ = Function(R)
        us_.assign(0.0)
        dx_solid = Measure("dx", domain=solid_mesh)

        Lm = TrialFunction(Z)
        e = TestFunction(Z)
        uf_ = Function(R)

        Lm_ = [Function(Z), Function(Z)]
        Lm_[0].assign(0.0)
        Lm_[1].assign(0.0)

        # ==================================
        # BOUNDARY CONDITIONS
        # ==================================
        FS = {'fluid': [W.sub(0), W.sub(1), Z1], 'lagrange': [Z]}
        if u_ex_val is not None:
            bcs = [
                DirichletBC(FS['fluid'][0], u_ex_val, 1),
                DirichletBC(FS['fluid'][0], u_ex_val, 3),
                DirichletBC(FS['fluid'][0], u_ex_val, 4)
            ]
            if g_ex_val is None:
                bcs.append(DirichletBC(FS['fluid'][0], u_ex_val, 2))
                if p_ex_val is not None:
                    bcs.append(DirichletBC(FS['fluid'][1], p_ex_val, 2))
            bcs_correction = [
                DirichletBC(V, u_ex_val, 1),
                DirichletBC(V, u_ex_val, 3),
                DirichletBC(V, u_ex_val, 4)
            ]
            if g_ex_val is None:
                bcs_correction.append(DirichletBC(V, u_ex_val, 2))
        else:
            bcs = create_boundary_conditions(fluid_mesh, type_obstacle=self.type_obstacle, **FS)
            bcs_correction = create_boundary_conditions_correction(fluid_mesh, V, type_obstacle=self.type_obstacle)

        # ==================================
        # FSI DELTA-INTERPOLATION
        # ==================================
        fsi_interpolation = FSIInterpolation()
        fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
        fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

        # ==================================
        # VARIATIONAL FORMULATIONS (3 Steps)
        # ==================================
        # Step 1: Tentative Stokes problem (DLM-Stokes-S1)
        a1 = 2.0 * Constant(mu) * inner(sym(grad(u)), sym(grad(v))) * dx_fluid \
            - div(v) * p * dx_fluid \
            + div(u) * q * dx_fluid

        L1 = inner(f, v) * dx_fluid \
            - inner(Lm_f, v) * dx_fluid

        if g_ex_val is not None:
            ds_b = Measure("ds", domain=fluid_mesh.mesh)
            L1 += inner(g_ex_val, v) * ds_b(2)

        # Step 2: Lagrange multiplier on solid mesh (DLM-Stokes-S2)
        a2 = inner(Lm, e) * dx_solid
        L2 = (Constant(rho) / Constant(dt_virtual)) * inner(uf_ - us_, e) * dx_solid \
            + inner(Lm_[1], e) * dx_solid

        # Step 3: Velocity correction on fluid mesh (DLM-Stokes-S3)
        u_v = TrialFunction(V)
        v_v = TestFunction(V)
        uh = Function(V)

        a3 = (Constant(rho) / Constant(dt_virtual)) * inner(u_v, v_v) * dx_fluid
        L3 = (Constant(rho) / Constant(dt_virtual)) * inner(u_star, v_v) * dx_fluid \
            - inner(Lm_f - Lm_f_old, v_v) * dx_fluid

        # ==================================
        # EXECUTE 3-STEP DLM SOLVE
        # ==================================
        # Step 1: Solve tentative velocity and pressure
        solve(a1 == L1, sol_star, bcs=bcs, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        })

        # Interpolate fluid velocity onto solid mesh
        uf_.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, u_star, "S"))

        # Step 2: Solve Lagrange multiplier on solid domain
        solve(a2 == L2, Lm_[0], solver_parameters={'ksp_type': 'bcgs', 'pc_type': 'sor'})

        # Interpolate Lagrange multiplier back to fluid domain
        Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[0], "F"))

        # Step 3: Solve velocity correction
        solve(a3 == L3, uh, bcs=bcs_correction, solver_parameters={'ksp_type': 'cg', 'pc_type': 'sor'})

        return uh, ph


# Helper function for quick warm-start initialization
def solve_stokes_dlm_initial(fluid_mesh=None, solid_mesh=None, obstacle=None, type_obstacle="square", n=None, Re=None, u_exact=None, p_exact=None, f_custom=None, g_custom=None):
    solver = Stokes_DLM_Solver(type_obstacle=type_obstacle, n=n, Re=Re)
    return solver.Stokes_DLM_Solve(fluid_mesh=fluid_mesh, solid_mesh=solid_mesh, obstacle=obstacle, f_custom=f_custom, u_exact=u_exact, p_exact=p_exact, g_custom=g_custom)
