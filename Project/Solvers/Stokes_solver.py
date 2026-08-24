import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import user_inputs.user_parameters as user_parameters


class Stokes_Solver:
    """
    Standard Stokes Solver used across all Navier-Stokes solvers
    to compute a smooth, divergence-free (div(u)=0) initial velocity field
    satisfying the prescribed boundary conditions and forcing at t = 0.
    """
    def __init__(self, mu=None, Re=None, rho=1.0, L_char=1.0, u_char=1.0):
        self.rho = rho
        self.u_char = u_char
        self.L_char = L_char
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        if mu is not None:
            self.mu = mu
        else:
            self.mu = self.rho * self.L_char * self.u_char / self.Re

    def solve_initial_velocity(self, mesh, bcs, f_custom=None, g_custom=None, W=None):
        """
        Solves:
            a(u, v) - (p, div v) + (div u, q) = (f, v) + <g, v>_Gamma2
        with direct MUMPS LU factorization.
        Returns:
            (uh, ph) as Function on velocity and pressure spaces.
        """
        # Unwrap domain mesh if wrapped object passed
        domain_mesh = mesh.mesh if hasattr(mesh, 'mesh') else mesh

        if W is None:
            V = VectorFunctionSpace(domain_mesh, 'P', 2)
            Q = FunctionSpace(domain_mesh, 'P', 1)
            W = V * Q
        else:
            V = W.sub(0)
            Q = W.sub(1)

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        sol = Function(W)
        uh, ph = sol.subfunctions

        dx_m = Measure("dx", domain=domain_mesh)

        f_val = f_custom(domain_mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0.0, 0.0)))
        g_val = g_custom(domain_mesh) if callable(g_custom) else g_custom

        a_init = 2.0 * Constant(self.mu) * inner(sym(grad(u)), sym(grad(v))) * dx_m \
                 - div(v) * p * dx_m \
                 + div(u) * q * dx_m

        L_init = inner(f_val, v) * dx_m
        if g_val is not None:
            ds_b = Measure("ds", domain=domain_mesh)
            L_init += inner(g_val, v) * ds_b(2)

        solve(a_init == L_init, sol, bcs=bcs, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        })

        uh_out = Function(W.sub(0))
        ph_out = Function(W.sub(1))
        uh_out.assign(uh)
        ph_out.assign(ph)

        return uh_out, ph_out


def solve_stokes_initial(mesh, bcs, mu=None, Re=None, f_custom=None, g_custom=None, W=None):
    """
    Unified entry-point for Stokes initialization called identically
    by all Navier-Stokes solvers (Conforming, DLM, Brinkman).
    """
    solver = Stokes_Solver(mu=mu, Re=Re)
    return solver.solve_initial_velocity(mesh=mesh, bcs=bcs, f_custom=f_custom, g_custom=g_custom, W=W)
