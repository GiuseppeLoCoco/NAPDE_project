"""
Numerical Experiment: Upstream Buffer Layer Dirichlet Recovery via Brinkman Penalization
Phase 1 (Conforming Benchmark on Omega_0) and Phase 2 (Buffer Recovery via Brinkman Penalization).

"""

import os
import sys
import math
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# Ensure Project and related directories are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
validation_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(validation_dir)
for p in [project_dir, validation_dir, os.path.join(project_dir, "domain_settings"),
         os.path.join(project_dir, "Utils"), os.path.join(project_dir, "Solvers")]:
    if p not in sys.path:
        sys.path.append(p)

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, sqrt,
    assemble, conditional, lt, ge, sin, cos, pi, Identity
)

from domain_settings.obstacles import BufferObstacle
from domain_settings.mesh_settings import unstructured_rectangle_mesh
from Utils.mms import ManufacturedSolution
from Solvers.NS_Brinkman import Brinkman_solver
from Solvers.NS_Conforming import Conforming_solver
from validation.checkpoint_loader import load_conforming_solution, load_brinkman_solution



# =============================================================================
# 2. PHASE 1: CONFORMING BENCHMARK SOLVER (Dominio Fisico Omega_0)
# =============================================================================

def solve_phase1_conforming(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                            T_end: float = 5.0, dt: float = 0.5, structured: bool = True):
    """
    Solve the NS problem on physical domain Omega_0 = [0, Lx] x [0, Ly] using the Conforming_solver
    with exact Dirichlet boundary conditions, loading checkpoint if available.
    """
    # 1. Check if checkpoint already exists
    mesh_chk, uh_chk, ph_chk = load_conforming_solution(
        obstacle_type=None, n=n, Re=mms.Re, t_final=T_end, is_mms=True
    )
    if mesh_chk is not None and uh_chk is not None and ph_chk is not None:
        return uh_chk, ph_chk, mesh_chk

    if structured:
        ny = max(4, int(round(n * Ly / Lx)))
        mesh = RectangleMesh(n, ny, Lx, Ly)
    else:
        mesh = unstructured_rectangle_mesh(0.0, Lx, 0.0, Ly, n, Ly_ref=Ly)

    solver = Conforming_solver(moving=False, type_obstacle=None, n=n, Re=mms.Re, structured=structured)
    mesh_out, uh, ph = solver.conforming_solve(
        mesh=mesh,
        obstacle=None,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        g_custom=mms.g_exact,
        u_init=None,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 3. PHASE 2: BUFFER RECOVERY SOLVER (Omega_buf + Omega_0 with Brinkman)
# =============================================================================

def solve_phase2_brinkman_buffer(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                                 L_buf: float = 1.0, R_penalty: float = 1.0e5,
                                 T_end: float = 5.0, dt: float = 0.5, structured: bool = True):
    """
    Solves extended problem on [-L_buf, Lx] x [0, Ly] using Brinkman_solver with BufferObstacle,
    loading checkpoint if available.
    """
    # 1. Check if checkpoint already exists
    mesh_chk, uh_chk, ph_chk = load_brinkman_solution(
        obstacle_type="buffer", n=n, R_val=R_penalty, Re=mms.Re, t_final=T_end, is_mms=True
    )
    if mesh_chk is not None and uh_chk is not None and ph_chk is not None:
        return uh_chk, ph_chk, mesh_chk

    if structured:
        nx_phys = n
        nx_buf = max(1, int(round(n * L_buf / Lx)))
        n_tot = nx_buf + nx_phys
        ny = max(4, int(round(n * Ly / Lx)))
        L_tot = L_buf + Lx

        mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
        mesh.coordinates.dat.data[:, 0] -= L_buf
    else:
        mesh = unstructured_rectangle_mesh(-L_buf, Lx, 0.0, Ly, n, Ly_ref=Ly)

    buf_obstacle = BufferObstacle(L_buf=L_buf)
    solver = Brinkman_solver(moving=False, type_obstacle="buffer", n=n, R=R_penalty, Re=mms.Re, structured=structured)

    mesh_out, uh, ph = solver.Brinkman_solve(
        mesh=mesh,
        obstacle=buf_obstacle,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        g_custom=mms.g_exact,
        u_init=None,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 4. ERROR EVALUATION & INTERFACE EXTRACTION
# =============================================================================

def compute_errors_phase1(mesh, uh, ph, mms: ManufacturedSolution, x_start: float = 0.0) -> Tuple[float, float, float]:
    """Compute the errors L2(u), H1(u) and L2(p) on Omega_0 (or [x_start, Lx] if x_start > 0)."""
    X = SpatialCoordinate(mesh)
    x = X[0]
    u_ex = mms.u_exact(mesh)
    p_ex = mms.p_exact(mesh)

    mask = conditional(ge(x, x_start), 1.0, 0.0) if x_start > 0.0 else Constant(1.0)

    err_u = uh - u_ex
    err_L2_u = sqrt(assemble(mask * inner(err_u, err_u) * dx(domain=mesh)))
    err_H1_u = sqrt(assemble(mask * (inner(err_u, err_u) + inner(grad(err_u), grad(err_u))) * dx(domain=mesh)))

    vol = assemble(mask * dx(domain=mesh))
    mean_ph = assemble(mask * ph * dx(domain=mesh)) / vol
    mean_pex = assemble(mask * p_ex * dx(domain=mesh)) / vol
    err_p = (ph - mean_ph) - (p_ex - mean_pex)
    err_L2_p = sqrt(assemble(mask * inner(err_p, err_p) * dx(domain=mesh)))

    return float(err_L2_u), float(err_H1_u), float(err_L2_p)


def compute_errors_phase2_restricted(mesh, uh, ph, mms: ManufacturedSolution, x_start: float = 0.0) -> Tuple[float, float, float]:
    """Compute the errors L2(u), H1(u) e L2(p) on Omega_0 restricted to x >= x_start."""
    X = SpatialCoordinate(mesh)
    x = X[0]
    u_ex = mms.u_exact(mesh)
    p_ex = mms.p_exact(mesh)

    mask_phys = conditional(ge(x, x_start), 1.0, 0.0)

    err_u = uh - u_ex
    err_L2_u = sqrt(assemble(mask_phys * inner(err_u, err_u) * dx(domain=mesh)))
    err_H1_u = sqrt(assemble(mask_phys * (inner(err_u, err_u) + inner(grad(err_u), grad(err_u))) * dx(domain=mesh)))

    vol_phys = assemble(mask_phys * dx(domain=mesh))
    mean_ph = assemble(mask_phys * ph * dx(domain=mesh)) / vol_phys
    mean_pex = assemble(mask_phys * p_ex * dx(domain=mesh)) / vol_phys
    err_p = (ph - mean_ph) - (p_ex - mean_pex)
    err_L2_p = sqrt(assemble(mask_phys * inner(err_p, err_p) * dx(domain=mesh)))

    return float(err_L2_u), float(err_H1_u), float(err_L2_p)


def compute_error_decay_slices(mesh, uh, mms: ManufacturedSolution, x_cuts: Optional[List[float]] = None) -> List[Tuple[float, float]]:
    """
    Computes L2(u) error restricted to [x_cut, Lx] for multiple cutoff positions x_cut >= 0.
    Useful to verify how quickly the Brinkman error decays as we move away from the interface x = 0.
    """
    if x_cuts is None:
        x_cuts = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    X = SpatialCoordinate(mesh)
    x = X[0]
    u_ex = mms.u_exact(mesh)
    err_u = uh - u_ex
    decay_results = []
    for xc in x_cuts:
        mask = conditional(ge(x, xc), 1.0, 0.0)
        err_val = float(sqrt(assemble(mask * inner(err_u, err_u) * dx(domain=mesh))))
        decay_results.append((xc, err_val))
    return decay_results


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
                val = uh.at(pt, tolerance=1e-4)
                u_num_x[i] = val[0]
            except Exception:
                try:
                    val = uh.at([1e-6, y_val], tolerance=1e-4)
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
    resolutions: List[int] = [75, 100, 125],
    Lx: float = 4.0,
    Ly: float = 1.0,
    L_buf: float = 1.0,
    Re: float = 40.0,
    R_penalty: float = 1.0e4,
    T_end: float = 5.0,
    dt: float = 0.5,
    x_start: float = 0.0,
    structured: bool = True,
    output_dir: str = "buffer_experiment_results"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    domain_str = f"Physical [{x_start}, {Lx}] x [0, {Ly}] (offset x_start = {x_start})" if x_start > 0.0 else f"Physical [0, {Lx}] x [0, {Ly}]"
    print("=" * 80)
    print("UPSTREAM BUFFER RECOVERY EXPERIMENT: CONFORMING vs BRINKMAN BUFFER")
    print(f"Domain: {domain_str} | Buffer length: {L_buf} | Re: {Re} | R: {R_penalty:.1e} | Structured: {structured}")
    print(f"Resolutions n: {resolutions} | Final Time T: {T_end}s (dt = {dt}s)")
    print("=" * 80)

    res_p1 = {"L2_u": [], "H1_u": [], "L2_p": [], "interf_L2": []}
    res_p2 = {"L2_u": [], "H1_u": [], "L2_p": [], "interf_L2": []}
    
    profiles_p2 = {}
    h_vals = [Lx / n for n in resolutions]

    # --- Loop of simulation on every resolution ---
    for n in resolutions:
        print(f"\n---> Running Resolution n = {n} (h = {Lx/n:.4f}, structured = {structured})")
        
        # 1. Phase 1: Conforming
        uh_1, ph_1, mesh_1 = solve_phase1_conforming(n, mms, Lx, Ly, T_end, dt, structured=structured)
        e_L2_u1, e_H1_u1, e_L2_p1 = compute_errors_phase1(mesh_1, uh_1, ph_1, mms, x_start=x_start)
        y_pts_1, u_num_x1, u_ex_x1 = extract_interface_profile(uh_1, mms)
        e_interf_L2_1 = float(np.sqrt(_trapezoid((u_num_x1 - u_ex_x1)**2, y_pts_1)))
        res_p1["L2_u"].append(e_L2_u1)
        res_p1["H1_u"].append(e_H1_u1)
        res_p1["L2_p"].append(e_L2_p1)
        res_p1["interf_L2"].append(e_interf_L2_1)
        label_p1 = f"[Phase 1 Conforming (x >= {x_start})]" if x_start > 0.0 else "[Phase 1 Conforming]"
        print(f"  {label_p1} L2(u): {e_L2_u1:.4e} | H1(u): {e_H1_u1:.4e} | L2(p): {e_L2_p1:.4e} | Intf_L2(x=0): {e_interf_L2_1:.4e}")

        # 2. Phase 2: Buffer Brinkman
        uh_2, ph_2, mesh_2 = solve_phase2_brinkman_buffer(n, mms, Lx, Ly, L_buf, R_penalty, T_end, dt, structured=structured)
        e_L2_u2, e_H1_u2, e_L2_p2 = compute_errors_phase2_restricted(mesh_2, uh_2, ph_2, mms, x_start=x_start)
        
        # Interface profile
        y_pts, u_num_x, u_ex_x = extract_interface_profile(uh_2, mms)
        profiles_p2[n] = (y_pts, u_num_x)
        e_interf_L2 = np.sqrt(_trapezoid((u_num_x - u_ex_x)**2, y_pts))
        
        res_p2["L2_u"].append(e_L2_u2)
        res_p2["H1_u"].append(e_H1_u2)
        res_p2["L2_p"].append(e_L2_p2)
        res_p2["interf_L2"].append(e_interf_L2)
        label_p2 = f"[Phase 2 Buffer Rec (x >= {x_start})]" if x_start > 0.0 else "[Phase 2 Buffer Rec]"
        print(f"  {label_p2} L2(u): {e_L2_u2:.4e} | H1(u): {e_H1_u2:.4e} | L2(p): {e_L2_p2:.4e} | Intf_L2(x=0): {e_interf_L2:.4e}")

        # Error decay analysis away from interface (x >= 0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
        decay_slices = compute_error_decay_slices(mesh_2, uh_2, mms)
        ref_err = decay_slices[0][1] if decay_slices and decay_slices[0][1] > 0 else 1.0
        decay_str = " | ".join([f"x>={xc:.2f}: {err:.3e} ({err/ref_err*100:5.1f}%)" for xc, err in decay_slices])
        print(f"  [Error Decay vs Distance from Interface x=0]\n    {decay_str}")

    # -------------------------------------------------------------------------
    # 5. PRINT CONVERGENCE SUMMARY TABLES (via experiment_plots module)
    # -------------------------------------------------------------------------
    from experiment_plots import (
        print_phase_comparison_tables,
        plot_phase_comparison_loglog,
        plot_interface_velocity_profile
    )

    method_title = f"Brinkman (x >= {x_start})" if x_start > 0.0 else "Brinkman"
    print_phase_comparison_tables(
        resolutions=resolutions,
        h_vals=h_vals,
        res_p1=res_p1,
        res_p2=res_p2,
        method_name=method_title
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
        method_name=method_title,
        output_path=conv_plot_path
    )

    # Figure 2: Recovery of the velocity profile x = 0
    profile_plot_path = os.path.join(output_dir, "interface_velocity_recovery.png")
    brinkman_labels = [f"Brinkman Rec. (n={n})" for n in resolutions]
    plot_interface_velocity_profile(
        profiles=profiles_p2,
        Ly=Ly,
        keys=resolutions,
        title="Velocity Profile Recovery at Interface $\\Sigma$ ($x = 0$)",
        output_path=profile_plot_path,
        custom_labels=brinkman_labels,
        xlim=(0.8, 1.2)
    )


# =============================================================================
# 7. EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_experiment_pipeline(
        resolutions=[40, 80, 120, 160],       # Resolutions
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,                      # Length of the buffer region
        Re=40.0,
        R_penalty=1.0e3,                # Brinkman penalty term
        T_end=30,                       # Final time
        dt=0.5,
        x_start=2,                    # Calcolo dell'errore a partire da x >= 0.1 (dopo l'interfaccia x=0)
        structured=False,               # Set False for unstructured mesh
        output_dir="results_Brinkman_buffer_recovery_unstructured"
    )