"""
Numerical Experiment: Upstream Buffer Layer Dirichlet Recovery via RIIS (Resistive Immersed Interface Solver)
Phase 1 (Conforming Benchmark on Omega_0) and Phase 2 (Buffer Recovery via RIIS).

"""

import os
import sys
import math
import warnings
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Ensure Project and related directories are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
for p in [project_dir, os.path.join(project_dir, "domain_settings"),
         os.path.join(project_dir, "Utils"), os.path.join(project_dir, "Solvers")]:
    if p not in sys.path:
        sys.path.append(p)

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, sqrt,
    assemble, conditional, lt, ge, sin, cos, pi
)

from domain_settings.obstacles import BufferObstacle
from Solvers.NS_RIIS import RIIS_solver
from Solvers.NS_Conforming import Conforming_solver


# =============================================================================
# 1. MMS EXACT SOLUTION & FORCING DEFINITION
# =============================================================================

class ManufacturedSolution:
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0, L_char: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = L_char  # Characteristic length for Reynolds number matching channel/buffer (1.0)
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
        """Compute the force of the exact solution for the Navier-Stokes problem"""
        X = SpatialCoordinate(mesh)
        u_ex = self.u_exact(mesh)
        p_ex = self.p_exact(mesh)
        
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press



# =============================================================================
# 2. PHASE 1: CONFORMING BENCHMARK SOLVER (Dominio Fisico Omega_0)
# =============================================================================

def solve_phase1_conforming(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                            T_end: float = 5.0, dt: float = 0.5):
    """
    Solve the NS problem on physical domain Omega_0 = [0, Lx] x [0, Ly] using the Conforming_solver
    with exact Dirichlet boundary conditions.
    """
    ny = max(4, int(round(n * Ly / Lx)))
    mesh = RectangleMesh(n, ny, Lx, Ly)

    solver = Conforming_solver(moving=False, type_obstacle=None, n=n, Re=mms.Re)
    mesh_out, uh, ph = solver.conforming_solve(
        mesh=mesh,
        obstacle=None,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        u_init=mms.u_exact,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 3. PHASE 2: BUFFER RECOVERY SOLVER (Omega_buf + Omega_0 with RIIS)
# =============================================================================

def solve_phase2_riis_buffer(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                             L_buf: float = 1.0, R_penalty: float = 1.0e5,
                             T_end: float = 5.0, dt: float = 0.5, eps: float = None):
    """
    Solves extended problem on [-L_buf, Lx] x [0, Ly] using RIIS_solver with BufferObstacle.
    """
    nx_phys = n
    nx_buf = max(1, int(round(n * L_buf / Lx)))
    n_tot = nx_buf + nx_phys
    ny = max(4, int(round(n * Ly / Lx)))
    L_tot = L_buf + Lx

    mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
    mesh.coordinates.dat.data[:, 0] -= L_buf

    eps_val = eps if eps is not None else (8.0 / n)
    buf_obstacle = BufferObstacle(L_buf=L_buf, riis_epsilon=eps_val)
    solver = RIIS_solver(moving=False, type_obstacle="buffer", n=n, R=R_penalty, Re=mms.Re, eps=eps_val)

    mesh_out, uh, ph = solver.RIIS_solve(
        mesh=mesh,
        obstacle=buf_obstacle,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        u_init=mms.u_exact,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 4. ERROR EVALUATION & INTERFACE EXTRACTION
# =============================================================================

def compute_errors_phase1(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Compute the errors L2(u), H1(u) and L2(p) on Omega_0."""
    u_ex = mms.u_exact(mesh)
    p_ex = mms.p_exact(mesh)

    err_u = uh - u_ex
    err_L2_u = sqrt(assemble(inner(err_u, err_u) * dx(domain=mesh)))
    err_H1_u = sqrt(assemble((inner(err_u, err_u) + inner(grad(err_u), grad(err_u))) * dx(domain=mesh)))

    vol = assemble(Constant(1.0) * dx(domain=mesh))
    mean_ph = assemble(ph * dx(domain=mesh)) / vol
    mean_pex = assemble(p_ex * dx(domain=mesh)) / vol
    err_p = (ph - mean_ph) - (p_ex - mean_pex)
    err_L2_p = sqrt(assemble(inner(err_p, err_p) * dx(domain=mesh)))

    return float(err_L2_u), float(err_H1_u), float(err_L2_p)


def compute_errors_phase2_restricted(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Compute the errors L2(u), H1(u) e L2(p) only on Omega_0 (x >= 0)."""
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
    """Extract the vertical profile of velocity u_x along the interface Sigma (x = 0)."""
    y_coords = np.linspace(0.0, mms.Ly, num_points)
    u_num_x = np.zeros(num_points)
    u_exact_x = np.zeros(num_points)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i, y_val in enumerate(y_coords):
            pt = [0.0, y_val]
            try:
                val = uh.at(pt, tolerance=1e-5)
                u_num_x[i] = val[0]
            except Exception:
                u_num_x[i] = 0.0

            # Exact analytical value: sin(pi * 0 / Lx) * sin(2*pi*y / Ly) = 0
            u_exact_x[i] = 1.0 + math.sin(0.0) * math.sin(2.0 * math.pi * y_val / mms.Ly)

    return y_coords, u_num_x, u_exact_x


# =============================================================================
# 5. PIPELINE for ANALYSIS and COMPARISON
# =============================================================================

def run_experiment_pipeline(
    resolutions: List[int] = [40, 80, 120],
    Lx: float = 4.0,
    Ly: float = 1.0,
    L_buf: float = 1.0,
    Re: float = 40.0,
    R_penalty: float = 1.0e4,
    T_end: float = 5.0,
    dt: float = 0.5,
    output_dir: str = "results_RIIS_buffer_recovery"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    print("=" * 80)
    print("UPSTREAM BUFFER RECOVERY EXPERIMENT: CONFORMING vs RIIS BUFFER")
    print(f"Domain: Physical [0, {Lx}] x [0, {Ly}] | Buffer length: {L_buf} | Re: {Re} | R: {R_penalty:.1e}")
    print(f"Resolutions n: {resolutions} | Final Time T: {T_end}s (dt = {dt}s)")
    print("=" * 80)

    res_p1 = {"L2_u": [], "H1_u": [], "L2_p": []}
    res_p2 = {"L2_u": [], "H1_u": [], "L2_p": [], "interf_L2": []}
    
    profiles_p2 = {}
    h_vals = [Lx / n for n in resolutions]

    # --- Loop of simulation on every resolution ---
    for n in resolutions:
        print(f"\n---> Running Resolution n = {n} (h = {Lx/n:.4f})")
        
        # 1. Phase 1: Conforming
        uh_1, ph_1, mesh_1 = solve_phase1_conforming(n, mms, Lx, Ly, T_end, dt)
        e_L2_u1, e_H1_u1, e_L2_p1 = compute_errors_phase1(mesh_1, uh_1, ph_1, mms)
        res_p1["L2_u"].append(e_L2_u1)
        res_p1["H1_u"].append(e_H1_u1)
        res_p1["L2_p"].append(e_L2_p1)
        print(f"  [Phase 1 Conforming] L2(u): {e_L2_u1:.4e} | H1(u): {e_H1_u1:.4e} | L2(p): {e_L2_p1:.4e}")

        # 2. Phase 2: Buffer RIIS
        uh_2, ph_2, mesh_2 = solve_phase2_riis_buffer(n, mms, Lx, Ly, L_buf, R_penalty, T_end, dt)
        e_L2_u2, e_H1_u2, e_L2_p2 = compute_errors_phase2_restricted(mesh_2, uh_2, ph_2, mms)
        
        # Interface profile
        y_pts, u_num_x, u_ex_x = extract_interface_profile(uh_2, mms)
        profiles_p2[n] = (y_pts, u_num_x)
        e_interf_L2 = np.sqrt(_trapezoid((u_num_x - u_ex_x)**2, y_pts))
        
        res_p2["L2_u"].append(e_L2_u2)
        res_p2["H1_u"].append(e_H1_u2)
        res_p2["L2_p"].append(e_L2_p2)
        res_p2["interf_L2"].append(e_interf_L2)
        print(f"  [Phase 2 Buffer Rec] L2(u): {e_L2_u2:.4e} | H1(u): {e_H1_u2:.4e} | L2(p): {e_L2_p2:.4e} | Intf_L2(x=0): {e_interf_L2:.4e}")

    # -------------------------------------------------------------------------
    # 5. PRINT CONVERGENCE SUMMARY TABLES (via experiment_plots module)
    # -------------------------------------------------------------------------
    from experiment_plots import (
        print_phase_comparison_tables,
        plot_phase_comparison_loglog,
        plot_interface_velocity_profile
    )

    print_phase_comparison_tables(
        resolutions=resolutions,
        h_vals=h_vals,
        res_p1=res_p1,
        res_p2=res_p2,
        method_name="RIIS"
    )

    # -------------------------------------------------------------------------
    # 6. GENERATE COMPARISON PLOTS (via experiment_plots module)
    # -------------------------------------------------------------------------

    # Figure 1: Convergence plot Log-Log
    conv_plot_path = os.path.join(output_dir, "convergence_comparison_loglog.png")
    plot_phase_comparison_loglog(
        h_vals=h_vals,
        res_p1=res_p1,
        res_p2=res_p2,
        method_name="RIIS",
        output_path=conv_plot_path
    )

    # Figure 2: Recovery of the velocity profile x = 0
    profile_plot_path = os.path.join(output_dir, "interface_velocity_recovery.png")
    riis_labels = [f"RIIS Rec. (n={n})" for n in resolutions]
    plot_interface_velocity_profile(
        profiles=profiles_p2,
        Ly=Ly,
        keys=resolutions,
        title="Velocity Profile Recovery at Interface $\\Sigma$ ($x = 0$)",
        output_path=profile_plot_path,
        custom_labels=riis_labels,
        xlim=(0.8, 1.2)
    )


# =============================================================================
# 7. EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_experiment_pipeline(
        resolutions=[40, 80, 120],       # Resolutions
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,                      # Length of the buffer region
        Re=40.0,
        R_penalty=1.0e6,                # RIIS penalty term R
        T_end=10.0,                     # Final time
        dt=0.5,
        output_dir="results_RIIS_buffer_recovery"
    )
