import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *

from user_inputs import *
import user_inputs.user_parameters as user_parameters
from domain_settings import create_bcs_penalty
from obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle

class Stokes_RIIS_solver:

    def __init__(self, type_obstacle="circle", n=None, R=None, Re=None):
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.R = R if R is not None else getattr(user_parameters, 'R', 1000.0)
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)

    def Stokes_RIIS_solve(self, mesh=None, W=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None):
        """
        Solves the stationary Stokes problem using RIIS (Regularized Interface Immersed Solid).
        Returns (uh, ph) as initial condition for velocity/pressure.
        """
        # ==================================
        # CREATE OBSTACLE & MESH (if not provided)
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

        if mesh is None:
            mesh = RectangleMesh(self.n, int(self.n * Ly / Lx), Lx, Ly)

        # ==================================
        # PARAMETERS & DATA
        # ==================================
        eps = 8.0 / self.n
        rho = 1.0
        u_char = 1.0
        L_char = self.obstacle.get_characteristic_length()
        mu = rho * L_char * u_char / self.Re
        R = self.R
        t = Constant(0.0)

        f = f_custom(mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        u_ex_val = u_exact(mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(mesh) if callable(g_custom) else g_custom

        # ==================================
        # FUNCTION SPACES
        # ==================================
        if W is None:
            V = VectorFunctionSpace(mesh, "CG", 2)
            Q = FunctionSpace(mesh, "CG", 1)
            W = V * Q

        # ==================================
        # BOUNDARY CONDITIONS
        # ==================================
        if u_ex_val is not None:
            bcs = [
                DirichletBC(W.sub(0), u_ex_val, 1), # Inflow
                DirichletBC(W.sub(0), u_ex_val, 3), # Bottom wall
                DirichletBC(W.sub(0), u_ex_val, 4)  # Top wall
            ]
            if g_ex_val is None:
                bcs.append(DirichletBC(W.sub(0), u_ex_val, 2)) # Outflow
                if p_ex_val is not None:
                    bcs.append(DirichletBC(W.sub(1), p_ex_val, 2))
        else:
            bcs = create_bcs_penalty(W, mesh, type_obstacle=self.type_obstacle)

        # ==================================
        # VARIATIONAL FORMULATION
        # ==================================
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        delta_expr = self.obstacle.delta(mesh, t)
        us_expr = u_ex_val if u_ex_val is not None else as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))

        a = 2.0 * Constant(mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
            - div(v) * p * dx \
            + div(u) * q * dx \
            + Constant(R / eps) * inner(u, v) * delta_expr * dx

        L = inner(f, v) * dx \
            + Constant(R / eps) * inner(us_expr, v) * delta_expr * dx

        if g_ex_val is not None:
            ds_b = Measure("ds", domain=mesh)
            L += inner(g_ex_val, v) * ds_b(2)

        # ==================================
        # SOLVE (Single Linear Step)
        # ==================================
        sol = Function(W)
        solve(a == L, sol, bcs=bcs, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        })

        uh, ph = sol.subfunctions
        return uh, ph


# Helper function for quick warm-start initialization
def solve_stokes_riis_initial(mesh=None, W=None, obstacle=None, type_obstacle="circle", n=None, Re=None, R=None, u_exact=None, p_exact=None, f_custom=None, g_custom=None):
    solver = Stokes_RIIS_solver(type_obstacle=type_obstacle, n=n, Re=Re, R=R)
    return solver.Stokes_RIIS_solve(mesh=mesh, W=W, obstacle=obstacle, f_custom=f_custom, u_exact=u_exact, p_exact=p_exact, g_custom=g_custom)
