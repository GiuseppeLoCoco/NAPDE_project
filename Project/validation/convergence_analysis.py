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
from typing import Dict, List, Tuple, Optional

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
from validation.checkpoint_loader import get_case_directory, get_field_filepath, load_hdf5_solution
from validation.validation_plots import plot_spatial_convergence_summary


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


def extract_vertical_profile(u_field, x_eval: float, Ly_val: float, num_points: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts velocity magnitude profile along a vertical line at x = x_eval.
    If a point falls inside a conforming mesh void, it defaults to 0.0 (no-slip rigid velocity).
    """
    y_coords = np.linspace(1e-5, Ly_val - 1e-5, num_points)
    u_mag = np.zeros_like(y_coords)

    for i, y in enumerate(y_coords):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                val = u_field.at([x_eval, y], tolerance=1e-6)
            u_mag[i] = math.sqrt(val[0]**2 + val[1]**2)
        except Exception:
            # Point is inside the void of a conforming mesh
            u_mag[i] = 0.0

    return y_coords, u_mag


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
    dt: Optional[float] = None,
    t_final: float = 20.0,
    t_final_conforming: Optional[float] = None
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

    t_conf = t_final_conforming if t_final_conforming is not None else t_final

    print("\n" + "=" * 75)
    print(f" STARTING CONVERGENCE ANALYSIS FOR {solver_type.upper()} ({obstacle_type.upper()}, Re={Re})")
    print(f" Refinements n: {resolutions}")
    print(f" Conforming reference n: {refinement_conforming} (at t = {t_conf:.2f}s)")
    if dt is not None:
        print(f" Time step dt: {dt}")
    print(f" Final time t_final ({solver_type.upper()}): {t_final:.2f}s")
    if solver_type.lower() == "brinkman":
        print(f" Brinkman Resistance R: {R_penalty}")
    print("=" * 75 + "\n")

    # -------------------------------------------------------------------------
    # STEP 1: CONFORMING REFERENCE SIMULATION
    # -------------------------------------------------------------------------
    conf_dir = get_case_directory("Conforming", obstacle_type, Re, refinement_conforming)
    conf_vel_file = get_field_filepath(conf_dir, "velocity", t_conf)
    conf_pres_file = get_field_filepath(conf_dir, "pressure", t_conf)

    if not os.path.exists(conf_vel_file):
        print(f">> Running Conforming solver for exact reference solution (n = {refinement_conforming}, t_final = {t_conf:.2f}s)...")
        conf_solver = Conforming_solver(moving=False, type_obstacle=obstacle_type, n=refinement_conforming, Re=Re)
        conf_solver.conforming_solve(dt=dt, t_final=t_conf)
        conf_dir = get_case_directory("Conforming", obstacle_type, Re, refinement_conforming)
        conf_vel_file = get_field_filepath(conf_dir, "velocity", t_conf)
        conf_pres_file = get_field_filepath(conf_dir, "pressure", t_conf)
    else:
        print(f">> Loaded existing Conforming reference solution (t = {t_conf:.2f}s) from: {conf_vel_file}")

    u_ex = load_hdf5_solution(conf_vel_file, "velocity")
    p_ex = load_hdf5_solution(conf_pres_file, "pressure") if os.path.exists(conf_pres_file) else None

    # Reference vertical profile
    y_ref, u_mag_ref = extract_vertical_profile(u_ex, x_obs, Ly)

    # -------------------------------------------------------------------------
    # STEP 2: LOOP FOR EACH RESOLUTION n (SIMULATION + ERROR CALCULATION)
    # -------------------------------------------------------------------------
    err_L2_u = []
    err_H1_u = []
    err_L2_p = []
    profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    R_values_used: Dict[int, float] = {}

    n_min = min(resolutions)

    for n in resolutions:

        print("\n" + "-" * 60)
        print(f" Running {solver_type} simulation for resolution n = {n} ...")
        print("-" * 60)

        # 1. Run specified solver
        if solver_type.lower() == "brinkman":
            solver = Brinkman_solver(moving=False, type_obstacle=obstacle_type, n=n, R=R_penalty, Re=Re)
            solver.Brinkman_solve(dt=dt, t_final=t_final)
            case_name = "Brinkman"
        elif solver_type.lower() in ("dlm", "ns_dlm"):
            solver = NS_DLM_Solver(moving=False, type_obstacle=obstacle_type, n=n, Re=Re)
            solver.NS_DLM_Solve(dt=dt, t_final=t_final)
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

        # 4. Vertical profile extraction
        profiles[n] = extract_vertical_profile(u_h, x_obs, Ly)

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
        p_str = f"{err_L2_p[i]:14.5e}" if not math.isnan(err_L2_p[i]) else "           N/A"
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
    # STEP 4: LOG-LOG CONVERGENCE PLOT, VERTICAL PROFILE AND SUMMARY TABLES
    # -------------------------------------------------------------------------
    case_name = "Brinkman" if solver_type.lower() == "brinkman" else "DLM"
    out_dir = os.path.join(project_dir, "Plots", case_name)
    os.makedirs(out_dir, exist_ok=True)
    plot_filename = os.path.join(out_dir, f"convergence_loglog_{case_name}_{obstacle_type}_Re{Re}.png")

    plot_spatial_convergence_summary(
        resolutions=resolutions,
        dx_h=dx_h,
        err_L2_u=err_L2_u,
        err_H1_u=err_H1_u,
        err_L2_p=err_L2_p,
        rates_L2_u=rates_L2_u,
        rates_H1_u=rates_H1_u,
        rates_L2_p=rates_L2_p,
        profiles=profiles,
        y_ref=y_ref,
        u_mag_ref=u_mag_ref,
        solver_type=solver_type,
        obstacle_type=obstacle_type,
        Re=Re,
        x_obs=x_obs,
        Ly=Ly,
        plot_filename=plot_filename
    )


# =============================================================================
# 4. SCRIPT EXECUTION AND PARAMETER SPECIFICATION
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # EDIT CONVERGENCE STUDY PARAMETERS HERE
    # =========================================================================
    resolutions = [40, 80, 120, 160]        # Mesh refinement levels n to simulate
    obstacle_type = "square"                     # "square" or "cylinder" (both stationary/fixed)
    solver_type = "Brinkman"                          # "Brinkman" or "dlm"
    Re = 40.0                                    # Reynolds number (can be ANY float/int, e.g. 40, 80, 100, 200...)
    refinement_conforming = 200                # Exact conforming reference mesh refinement
    R_penalty = 100000.0                         # Resistive parameter R (for Brinkman solver)
    dt = 0.5                                     # Time step size dt (can be None to use solver default)
    t_final = 40.0                               # Final simulation time step t_final for DLM / Brinkman
    t_final_conforming = 20.0                    # Reference Conforming time (can be different from t_final, e.g. 20.0 if already stationary)
    # =========================================================================

    run_convergence_analysis(
        resolutions=resolutions,
        obstacle_type=obstacle_type,
        solver_type=solver_type,
        Re=Re,
        refinement_conforming=refinement_conforming,
        R_penalty=R_penalty,
        dt=dt,
        t_final=t_final,
        t_final_conforming=t_final_conforming
    )
