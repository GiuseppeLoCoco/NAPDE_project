"""
Numerical Experiment - Strategy A:
Brinkman Penalty Parameter (R) Convergence Sweep at Fixed Mesh Resolution.

This script analyzes the asymptotic modeling error of the Brinkman upstream buffer
by varying R over several orders of magnitude on a fixed fine mesh (n = 100).
"""

import gc
import os
import sys
import math
import warnings
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Ensure Project and related directories are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
for p in [project_dir, os.path.join(project_dir, "domain_settings"),
         os.path.join(project_dir, "Utils"), os.path.join(project_dir, "Solvers")]:
    if p not in sys.path:
        sys.path.append(p)

# Compatibility for NumPy 1.x and 2.x trapezoidal integration
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, sqrt,
    assemble, conditional, lt, ge, sin, cos, pi
)

from domain_settings.obstacles import BufferObstacle
from Solvers.NS_Brinkman import Brinkman_solver


# =============================================================================
# 1. MMS EXACT SOLUTION & BODY FORCING
# =============================================================================

class ManufacturedSolution:
    """Exact divergence-free analytical solution and corresponding Navier-Stokes body force."""
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = 0.2
        self.mu = self.rho * self.L_char * self.u_char / self.Re

    def u_exact(self, mesh):
        X = SpatialCoordinate(mesh)
        x, y = X[0], X[1]
        u_x = 1.0 + sin(pi * x / self.Lx) * sin(2.0 * pi * y / self.Ly)
        u_y = (self.Ly / (2.0 * self.Lx)) * cos(pi * x / self.Lx) * (cos(2.0 * pi * y / self.Ly) - 1.0)
        return as_vector([u_x, u_y])

    def p_exact(self, mesh):
        X = SpatialCoordinate(mesh)
        x, y = X[0], X[1]
        return sin(pi * x / self.Lx) * sin(pi * y / self.Ly)

    def f_forcing(self, mesh):
        X = SpatialCoordinate(mesh)
        u_ex = self.u_exact(mesh)
        p_ex = self.p_exact(mesh)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press



# =============================================================================
# 2. BRINKMAN BUFFER SOLVER
# =============================================================================

def solve_brinkman_buffer(n: int, R_val: float, mms: ManufacturedSolution,
                          Lx: float = 4.0, Ly: float = 1.0, L_buf: float = 1.0,
                          T_end: float = 2.0, dt: float = 0.5) -> Tuple[object, object, object]:
    """
    Solves flow on the extended domain [-L_buf, Lx] x [0, Ly] with Brinkman penalization
    using the Brinkman_solver class.
    """
    nx_phys = n
    nx_buf = max(1, int(round(n * L_buf / Lx)))
    n_tot = nx_buf + nx_phys
    ny = max(4, int(round(n * Ly / Lx)))
    L_tot = L_buf + Lx

    mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
    mesh.coordinates.dat.data[:, 0] -= L_buf

    buf_obstacle = BufferObstacle(L_buf=L_buf)
    solver = Brinkman_solver(moving=False, n=n, R=R_val, Re=mms.Re)

    mesh_out, uh, ph = solver.Brinkman_solve(
        mesh=mesh,
        obstacle=buf_obstacle,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 3. ERROR POST-PROCESSING & PROFILE EXTRACTION
# =============================================================================

def compute_restricted_errors(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Computes L2(u), H1(u), and L2(p) error norms restricted strictly to Omega_0 (x >= 0)."""
    X = SpatialCoordinate(mesh)
    x = X[0]
    u_ex = mms.u_exact(mesh)
    p_ex = mms.p_exact(mesh)

    mask_phys = conditional(ge(x, 0.0), 1.0, 0.0)

    err_u = uh - u_ex
    err_L2_u = sqrt(assemble(mask_phys * inner(err_u, err_u) * dx(domain=mesh)))
    err_H1_u = sqrt(assemble(mask_phys * (inner(err_u, err_u) + inner(grad(err_u), grad(err_u))) * dx(domain=mesh)))

    vol_phys = assemble(mask_phys * dx(domain=mesh))
    mean_ph = assemble(mask_phys * ph * dx(domain=mesh)) / vol_phys
    mean_pex = assemble(mask_phys * p_ex * dx(domain=mesh)) / vol_phys
    err_p = (ph - mean_ph) - (p_ex - mean_pex)
    err_L2_p = sqrt(assemble(mask_phys * inner(err_p, err_p) * dx(domain=mesh)))

    return float(err_L2_u), float(err_H1_u), float(err_L2_p)


def extract_interface_profile(uh, mms: ManufacturedSolution, num_points: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts vertical cut of u_x at the physical interface Sigma (x = 0)."""
    y_coords = np.linspace(0.0, mms.Ly, num_points)
    u_num_x = np.zeros(num_points)
    u_exact_x = np.zeros(num_points)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i, y_val in enumerate(y_coords):
            try:
                val = uh.at([0.0, y_val], tolerance=1e-5)
                u_num_x[i] = val[0]
            except Exception:
                u_num_x[i] = 0.0
            u_exact_x[i] = 1.0 + math.sin(0.0) * math.sin(2.0 * math.pi * y_val / mms.Ly)

    return y_coords, u_num_x, u_exact_x


# =============================================================================
# 4. SWEEP EXECUTION & PLOTTING PIPELINE
# =============================================================================

def run_r_sweep_analysis(
    R_values: List[float] = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7],
    fixed_n: int = 100,
    Lx: float = 4.0,
    Ly: float = 1.0,
    L_buf: float = 1.0,
    Re: float = 40.0,
    T_end: float = 2.0,
    dt: float = 0.5,
    output_dir: str = "results_strategy_A_R_sweep"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    print("=" * 90)
    print("STRATEGY A: PENALTY PARAMETER SWEEP (R-SWEEP AT FIXED MESH RESOLUTION)")
    print(f"Fixed Mesh Resolution: n = {fixed_n} (h = {Lx/fixed_n:.4f}) | Re = {Re}")
    print(f"Brinkman Resistance Range R: {R_values}")
    print(f"Simulation Time: T = {T_end}s (dt = {dt}s)")
    print("=" * 90)

    errs_L2_u, errs_H1_u, errs_L2_p, errs_intf = [], [], [], []
    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for R_val in R_values:
        print(f"\n---> Solving for Brinkman Penalty R = {R_val:.1e} ...")
        uh, ph, mesh = solve_brinkman_buffer(fixed_n, R_val, mms, Lx, Ly, L_buf, T_end, dt)
        
        e_L2, e_H1, e_p = compute_restricted_errors(mesh, uh, ph, mms)
        y_pts, u_x_num, u_x_ex = extract_interface_profile(uh, mms)
        e_intf = float(np.sqrt(_trapezoid((u_x_num - u_x_ex)**2, y_pts)))

        errs_L2_u.append(e_L2)
        errs_H1_u.append(e_H1)
        errs_L2_p.append(e_p)
        errs_intf.append(e_intf)
        profiles[R_val] = (y_pts, u_x_num)

        print(f"     L2(u) in Omega_0: {e_L2:.5e} | H1(u): {e_H1:.5e} | Interface L2: {e_intf:.5e}")

        del uh, ph, mesh
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. PRINT CONVERGENCE SUMMARY TABLE (via experiment_plots module)
    # -------------------------------------------------------------------------
    from experiment_plots import (
        print_strategy_a_table,
        plot_strategy_a_r_sweep,
        plot_interface_velocity_profile
    )

    print_strategy_a_table(
        R_values=R_values,
        errs_L2_u=errs_L2_u,
        errs_intf=errs_intf
    )

    # -------------------------------------------------------------------------
    # 5. GENERATE COMPARISON PLOTS (via experiment_plots module)
    # -------------------------------------------------------------------------

    # Plot 1: Error vs R
    plot_conv = os.path.join(output_dir, "strategy_A_error_vs_R.png")
    plot_strategy_a_r_sweep(
        R_values=R_values,
        errs_L2_u=errs_L2_u,
        errs_H1_u=errs_H1_u,
        errs_L2_p=errs_L2_p,
        errs_intf=errs_intf,
        fixed_n=fixed_n,
        output_path=plot_conv
    )

    # Plot 2: Interface Velocity Profile vs R
    plot_prof = os.path.join(output_dir, "strategy_A_interface_profile.png")
    r_labels = [f"$R = 10^{{{int(math.log10(r))}}}$" for r in R_values]
    plot_interface_velocity_profile(
        profiles=profiles,
        Ly=Ly,
        keys=R_values,
        title=f"Interface Velocity Profile $u_x(0, y)$ vs $R$ ($n = {fixed_n}$)",
        output_path=plot_prof,
        custom_labels=r_labels
    )


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_r_sweep_analysis(
        R_values=[1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7],
        fixed_n=60,
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,
        Re=40.0,
        T_end=5.0,
        dt=0.2,
        output_dir="results_buffer_recovery_v2"
    )