"""
Convergence analysis script for stationary flow past obstacle (Square or Cylinder).

Features:
  - User-configurable parameters set directly at the top of the file / __main__ block.
  - Runs/checks Conforming reference simulation for refinement `refinement_conforming`.
  - Performs a `for n in resolutions:` loop executing the specified solver (Brinkman or DLM).
  - Loads the saved checkpoint solutions at final time t = T_end.
  - Calculates fluid-restricted velocity (L2, H1) errors and pressure L2 error.
  - Calculates empirical convergence rates and plots log-log error curves vs mesh size h = 1/n.
"""

import os
import sys
import math
import warnings
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from firedrake import (Constant, project, assemble, dx, inner, grad, sqrt,
                        CheckpointFile)

# Ensure Project directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from domain_settings.obstacles import circleObstacle, squareObstacle
from user_inputs.user_parameters import x_obs, y_obs, r_obs, side_length, Lx, Ly

from Solvers.NS_Conforming import Conforming_solver
from Solvers.NS_Brinkman import Brinkman_solver
from Solvers.NS_DLM_simple import NS_DLM_Solver


# =============================================================================
# 1. HELPER FUNCTIONS FOR PATH RESOLUTION AND HDF5 LOADING
# =============================================================================

def get_case_directory(solver_name: str, obstacle: str, Re: float, n: int, R_penalty: float = 1000.0) -> str:
    """Returns the output directory path where solver checkpoints are saved."""
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    param_str = f"n{n}_R{R_penalty}_Re{Re}" if solver_name.lower() == "brinkman" else f"n{n}_Re{Re}"
    return os.path.join(project_dir, "Plots", solver_name, "fixed", obstacle, sym_str, param_str)


def get_field_filepath(case_dir: str, field_name: str, t_val: float) -> str:
    """Finds the .h5 checkpoint filepath for a given field at time t_val."""
    return os.path.join(case_dir, field_name, f"{field_name}_t={t_val:.2f}.h5")


def load_hdf5_solution(filepath: str, field_name: str):
    """Loads Firedrake function from CheckpointFile .h5 file."""
    if not filepath.endswith(".h5"):
        filepath += ".h5"

    with CheckpointFile(filepath, 'r') as chk:
        mesh = chk.load_mesh()
        field = chk.load_function(mesh, field_name)
    return field


# =============================================================================
# 2. ERROR COMPUTATION RESTRICTED TO FLUID DOMAIN
# =============================================================================

def compute_fluid_errors_velocity(u_ex, u_h, obstacle_instance, t_val: float) -> Tuple[float, float]:
    """Computes L2 and H1 velocity error norms strictly inside the fluid domain (1 - chi)."""
    V_ex = u_ex.function_space()
    mesh_ex = V_ex.mesh()
    t_const = Constant(t_val)

    u_h_proj = project(u_h, V_ex)

    chi_solid = obstacle_instance.chi(mesh_ex, t_const)
    mask_fluid = 1.0 - chi_solid

    err = u_h_proj - u_ex

    err_L2_sq = assemble(mask_fluid * inner(err, err) * dx)
    err_H1_sq = assemble(mask_fluid * (inner(err, err) + inner(grad(err), grad(err))) * dx)

    return sqrt(err_L2_sq), sqrt(err_H1_sq)


def compute_fluid_error_pressure(p_ex, p_h, obstacle_instance, t_val: float) -> float:
    """Computes L2 pressure error norm in fluid domain (with mean pressure removed)."""
    V_ex = p_ex.function_space()
    mesh_ex = V_ex.mesh()
    t_const = Constant(t_val)

    p_h_proj = project(p_h, V_ex)

    chi_solid = obstacle_instance.chi(mesh_ex, t_const)
    mask_fluid = 1.0 - chi_solid

    vol_fluid = assemble(mask_fluid * dx(domain=mesh_ex))
    mean_p_ex = assemble(mask_fluid * p_ex * dx) / vol_fluid
    mean_p_h = assemble(mask_fluid * p_h_proj * dx) / vol_fluid

    err_p = (p_h_proj - mean_p_h) - (p_ex - mean_p_ex)
    err_p_L2_sq = assemble(mask_fluid * inner(err_p, err_p) * dx)

    return sqrt(err_p_L2_sq)


# =============================================================================
# 3. MAIN SIMULATION & CONVERGENCE LOOP
# =============================================================================

def run_convergence_analysis(
    resolutions: List[int],
    obstacle_type: str,
    solver_type: str,
    Re: float,
    refinement_conforming: int,
    R_penalty: float = 1000.0,
    t_final: float = 20.0
):
    """
    Executes simulations for each resolution n, loads saved results,
    computes L2/H1 errors, prints convergence rates, and plots log-log graphs.
    """

    # Instantiate obstacle object
    if obstacle_type.lower() == "square":
        obstacle_instance = squareObstacle(x_obs, y_obs, side_length)
    elif obstacle_type.lower() == "cylinder":
        obstacle_instance = circleObstacle(x_obs, y_obs, r_obs)
    else:
        raise ValueError(f"Unsupported obstacle type '{obstacle_type}'. Choose 'square' or 'cylinder'.")

    print("\n" + "=" * 75)
    print(f" STARTING CONVERGENCE ANALYSIS FOR {solver_type.upper()} ({obstacle_type.upper()}, Re={Re})")
    print(f" Refinements n: {resolutions}")
    print(f" Conforming reference n: {refinement_conforming}")
    if solver_type.lower() == "brinkman":
        print(f" Brinkman Resistance R: {R_penalty}")
    print("=" * 75 + "\n")

    # -------------------------------------------------------------------------
    # STEP 1: CONFORMING REFERENCE SIMULATION
    # -------------------------------------------------------------------------
    conf_dir = get_case_directory("Conforming", obstacle_type, Re, refinement_conforming)
    conf_vel_file = get_field_filepath(conf_dir, "velocity", t_final)
    conf_pres_file = get_field_filepath(conf_dir, "pressure", t_final)

    if not os.path.exists(conf_vel_file):
        print(f">> Running Conforming solver for exact reference solution (n = {refinement_conforming})...")
        conf_solver = Conforming_solver(moving=False, type_obstacle=obstacle_type, n=refinement_conforming, Re=Re)
        conf_solver.conforming_solve()
        conf_dir = get_case_directory("Conforming", obstacle_type, Re, refinement_conforming)
        conf_vel_file = get_field_filepath(conf_dir, "velocity", t_final)
        conf_pres_file = get_field_filepath(conf_dir, "pressure", t_final)
    else:
        print(f">> Loaded existing Conforming reference solution from: {conf_vel_file}")

    u_ex = load_hdf5_solution(conf_vel_file, "velocity")
    p_ex = load_hdf5_solution(conf_pres_file, "pressure") if os.path.exists(conf_pres_file) else None

    # -------------------------------------------------------------------------
    # STEP 2: LOOP FOR EACH RESOLUTION n (SIMULATION + ERROR CALCULATION)
    # -------------------------------------------------------------------------
    err_L2_u = []
    err_H1_u = []
    err_L2_p = []

    for n in resolutions:
        print("\n" + "-" * 60)
        print(f" Running {solver_type} simulation for resolution n = {n} ...")
        print("-" * 60)

        # 1. Run specified solver
        if solver_type.lower() == "brinkman":
            solver = Brinkman_solver(moving=False, type_obstacle=obstacle_type, n=n, R=R_penalty, Re=Re)
            solver.Brinkman_solve()
            case_name = "Brinkman"
        elif solver_type.lower() in ("dlm", "ns_dlm"):
            solver = NS_DLM_Solver(moving=False, type_obstacle=obstacle_type, n=n, Re=Re)
            solver.NS_DLM_Solve()
            case_name = "DLM"
        else:
            raise ValueError(f"Unknown solver type '{solver_type}'. Must be 'Brinkman' or 'dlm'.")

        # 2. Get output directory & load saved checkpoint data
        case_dir = get_case_directory(case_name, obstacle_type, Re, n, R_penalty)
        vel_file = get_field_filepath(case_dir, "velocity", t_final)
        pres_file = get_field_filepath(case_dir, "pressure", t_final)

        print(f" Loading solution at t = {t_final:.2f}s from: {vel_file}")
        u_h = load_hdf5_solution(vel_file, "velocity")

        # 3. Compute fluid errors
        e_L2_u, e_H1_u = compute_fluid_errors_velocity(u_ex, u_h, obstacle_instance, t_final)
        err_L2_u.append(float(e_L2_u))
        err_H1_u.append(float(e_H1_u))

        if p_ex is not None and os.path.exists(pres_file):
            p_h = load_hdf5_solution(pres_file, "pressure")
            e_L2_p = compute_fluid_error_pressure(p_ex, p_h, obstacle_instance, t_final)
            err_L2_p.append(float(e_L2_p))
        else:
            err_L2_p.append(float('nan'))

        print(f"   --> n={n:3d} | Error L2(u) = {e_L2_u:.5e} | Error H1(u) = {e_H1_u:.5e} | Error L2(p) = {err_L2_p[-1]:.5e}")

    # -------------------------------------------------------------------------
    # STEP 3: CONVERGENCE RATES COMPUTATION & SUMMARY
    # -------------------------------------------------------------------------
    dx_h = [1.0 / n for n in resolutions]
    rates_L2_u, rates_H1_u, rates_L2_p = [], [], []

    for i in range(len(resolutions) - 1):
        log_h = np.log(dx_h[i] / dx_h[i + 1])
        rates_L2_u.append(np.log(err_L2_u[i] / err_L2_u[i + 1]) / log_h)
        rates_H1_u.append(np.log(err_H1_u[i] / err_H1_u[i + 1]) / log_h)
        if not math.isnan(err_L2_p[i]) and not math.isnan(err_L2_p[i + 1]):
            rates_L2_p.append(np.log(err_L2_p[i] / err_L2_p[i + 1]) / log_h)
        else:
            rates_L2_p.append(float('nan'))

    print("\n" + "=" * 70)
    print(f" CONVERGENCE RESULTS SUMMARY ({solver_type.upper()}, {obstacle_type.upper()}, Re={Re})")
    print("=" * 70)
    print(f"{'n':>6} | {'h':>10} | {'L2 Velocity Err':>16} | {'H1 Velocity Err':>16} | {'L2 Pressure Err':>16}")
    print("-" * 70)
    for i, n in enumerate(resolutions):
        print(f"{n:6d} | {dx_h[i]:10.5f} | {err_L2_u[i]:16.5e} | {err_H1_u[i]:16.5e} | {err_L2_p[i]:16.5e}")
    print("-" * 70)

    print("\nEmpirical Convergence Rates:")
    for i in range(len(resolutions) - 1):
        n1, n2 = resolutions[i], resolutions[i + 1]
        print(f"  n = {n1} -> {n2}:  Rate L2(u) = {rates_L2_u[i]:+.3f} | "
              f"Rate H1(u) = {rates_H1_u[i]:+.3f} | "
              f"Rate L2(p) = {rates_L2_p[i]:+.3f}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # STEP 4: LOG-LOG CONVERGENCE PLOT AND SUMMARY TABLE IMAGE
    # -------------------------------------------------------------------------
    case_name = "Brinkman" if solver_type.lower() == "brinkman" else "DLM"
    out_dir = os.path.join(project_dir, "Plots", case_name)
    os.makedirs(out_dir, exist_ok=True)
    plot_filename = os.path.join(out_dir, f"convergence_loglog_{case_name}_{obstacle_type}_Re{Re}.png")

    fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [1.25, 1]})

    # 1. Log-Log Error Curves Plot
    ax_plot.loglog(dx_h, err_L2_u, 'o-', color='#2c7bb6', linewidth=2, markersize=8, label='$L^2$ Velocity Error')
    ax_plot.loglog(dx_h, err_H1_u, 's-', color='#d7191c', linewidth=2, markersize=8, label='$H^1$ Velocity Error')
    if not any(math.isnan(e) for e in err_L2_p):
        ax_plot.loglog(dx_h, err_L2_p, '^--', color='#2b83ba', linewidth=1.8, markersize=7, label='$L^2$ Pressure Error')

    # Reference slopes O(h) and O(h^2)
    h_arr = np.array(dx_h)
    ref_scale_1 = err_L2_u[0] / h_arr[0]
    ref_scale_2 = err_L2_u[0] / (h_arr[0] ** 2)
    ax_plot.loglog(h_arr, ref_scale_1 * h_arr, 'k--', alpha=0.6, label='$O(h)$')
    ax_plot.loglog(h_arr, ref_scale_2 * (h_arr ** 2), 'k:', alpha=0.6, label='$O(h^2)$')

    ax_plot.set_xlabel("Mesh size $h = 1/n$", fontsize=12)
    ax_plot.set_ylabel("Error Norm (at $t = T_{final}$)", fontsize=12)
    ax_plot.set_title(f"Spatial Convergence — {case_name} ({obstacle_type.capitalize()}, Re={Re})", fontsize=13, fontweight='bold')
    ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_plot.legend(fontsize=10, framealpha=0.9)

    # 2. Convergence Summary Tables
    ax_table.axis('off')
    ax_table.text(0.5, 0.98, f"Convergence Analysis Summary\n({case_name}, {obstacle_type.capitalize()}, Re={Re})",
                  fontsize=12, fontweight='bold', ha='center', va='top', transform=ax_table.transAxes)

    # Error Table
    err_headers = ['n', 'h', 'L²(u) Error', 'H¹(u) Error', 'L²(p) Error']
    err_rows = []
    for i, n_val in enumerate(resolutions):
        p_str = f"{err_L2_p[i]:.3e}" if not math.isnan(err_L2_p[i]) else "N/A"
        err_rows.append([f"{n_val}", f"{dx_h[i]:.4f}", f"{err_L2_u[i]:.3e}", f"{err_H1_u[i]:.3e}", p_str])

    table_err = ax_table.table(cellText=err_rows, colLabels=err_headers, loc='top', cellLoc='center', bbox=[0.0, 0.52, 1.0, 0.38])
    table_err.auto_set_font_size(False)
    table_err.set_fontsize(9)

    for (row, col), cell in table_err.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c7bb6')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f7f7f7')

    # Rates Table
    rate_headers = ['Interval (n)', 'Rate L²(u)', 'Rate H¹(u)', 'Rate L²(p)']
    rate_rows = []
    for i in range(len(resolutions) - 1):
        n1, n2 = resolutions[i], resolutions[i + 1]
        p_rate_str = f"{rates_L2_p[i]:+.3f}" if not math.isnan(rates_L2_p[i]) else "N/A"
        rate_rows.append([f"{n1} → {n2}", f"{rates_L2_u[i]:+.3f}", f"{rates_H1_u[i]:+.3f}", p_rate_str])

    table_rate = ax_table.table(cellText=rate_rows, colLabels=rate_headers, loc='bottom', cellLoc='center', bbox=[0.0, 0.05, 1.0, 0.35])
    table_rate.auto_set_font_size(False)
    table_rate.set_fontsize(9)

    for (row, col), cell in table_rate.get_celld().items():
        if row == 0:
            cell.set_facecolor('#d7191c')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f7f7f7')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f">> Convergence plot & table saved to: {plot_filename}")
    plt.close(fig)


# =============================================================================
# 4. SCRIPT EXECUTION AND PARAMETER SPECIFICATION
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # EDIT CONVERGENCE STUDY PARAMETERS HERE
    # =========================================================================
    resolutions = [50, 75, 100,125, 150]         # Mesh refinement levels n to simulate
    obstacle_type = "square"             # "square" or "cylinder" (both stationary/fixed)
    solver_type = "dlm"             # "Brinkman" or "dlm"
    Re = 40.0                            # Reynolds number (can be ANY float/int, e.g. 40, 80, 100, 200...)
    refinement_conforming = 200          # Exact conforming reference mesh refinement
    R_penalty = 100000.0                   # Resistive parameter R (for Brinkman solver)
    t_final = 20.0                       # Final simulation time step t_final
    # =========================================================================

    run_convergence_analysis(
        resolutions=resolutions,
        obstacle_type=obstacle_type,
        solver_type=solver_type,
        Re=Re,
        refinement_conforming=refinement_conforming,
        R_penalty=R_penalty,
        t_final=t_final
    )
