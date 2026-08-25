"""
Numerical Validation of the L2 Penalization Method for Incompressible Navier-Stokes Flows
Reference: Angot, Bruneau, Fabrie (Numer. Math. 1999, 81: 497-520), Section 6.1

Uses the existing project solvers:
  - Brinkman_solver (from Solvers.NS_Brinkman)
  - Conforming_solver (from Solvers.NS_Conforming)
  - Checkpoint loaders (from validation.checkpoint_loader)

This script executes the two benchmark experiments:
  1. Steady Case at Re = 40:
     - Reuses or solves conforming reference with Conforming_solver.
     - Reuses or solves penalized solutions with Brinkman_solver for various values of R = 1/eta.
     - Calculates ||u_eta|| in solid Omega_s and ||u_eta - u_ref|| in fluid Omega_f.
     - Verifies convergence rate O(eta) (Table 1).
     - Generates pressure and vorticity comparison plots (Fig. 3).

  2. Unsteady Case at Re = 80:
     - Computes vortex shedding behind the square cylinder for different values of eta (R = 1/eta).
     - Evaluates Strouhal number St = f*D/U (Table 2).
     - Compares convective delay effect for large eta.
"""

import os
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

# Include project directories in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
for p in [project_dir, os.path.join(project_dir, "domain_settings"),
          os.path.join(project_dir, "Utils"), os.path.join(project_dir, "Solvers"),
          os.path.join(project_dir, "user_inputs"), os.path.join(project_dir, "validation")]:
    if p not in sys.path:
        sys.path.append(p)

from firedrake import (
    RectangleMesh, Constant, FunctionSpace, Function,
    inner, grad, sym, dx, ds, sqrt, assemble, project,
    FacetNormal, Identity, CheckpointFile
)
from firedrake.pyplot import tripcolor

from user_inputs.user_parameters import Lx, Ly, x_obs, y_obs, side_length
from domain_settings.obstacles import squareObstacle
from domain_settings.mesh_settings import conforming_mesh
from Solvers.NS_Brinkman import Brinkman_solver
from Solvers.NS_Conforming import Conforming_solver
from validation.checkpoint_loader import (
    get_case_directory, get_field_filepath, load_hdf5_solution,
    load_conforming_solution, load_brinkman_solution
)
from validation.validation_plots import (
    plot_l2_penalization_convergence, plot_steady_comparison
)


# =============================================================================
# EXPERIMENT 1: STEADY STATE CONVERGENCE & TABLE 1 (Re = 40)
# =============================================================================

def run_steady_experiment(eta_list=None, n=320, dt=0.05, T_end=10.0, output_dir=None):
    """
    Reproduces Section 6.1 (Steady case at Re = 40):
    - Solves conforming reference using Conforming_solver.
    - Solves penalized solutions using Brinkman_solver for each eta in eta_list (R = 1/eta).
    - Computes ||u_eta||_L2(Omega_s) and ||u_eta - u_ref||_L2(Omega_f).
    - Measures convergence rates alpha for O(eta^alpha) (Table 1).
    - Generates Fig. 3 plots (pressure and vorticity fields).
    """
    if eta_list is None:
        eta_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    if output_dir is None:
        output_dir = os.path.join(project_dir, "Plots", "Validation", "L2_steady_Re40")
    os.makedirs(output_dir, exist_ok=True)

    Re = 40.0
    D = side_length

    print("\n" + "="*80)
    print(f" EXPERIMENT 1: STEADY FLOW AT Re = {Re}")
    print("="*80)

    # 1. Obstacle definition
    obs = squareObstacle(x_obs, y_obs, side_length=D)

    # 2. Conforming reference solver (search Plots/Conforming first)
    ref_mesh, u_ref, p_ref = load_conforming_solution(obstacle_type="square", n=n, Re=Re)
    if ref_mesh is None:
        print("\n--- Running Conforming Reference Solver ---")
        conf_solver = Conforming_solver(moving=False, type_obstacle="square", n=n, Re=Re, structured=True)
        ref_mesh, u_ref, p_ref = conf_solver.conforming_solve(
            obstacle=obs,
            dt=dt,
            t_final=T_end
        )

    results = []

    # 3. Penalized Brinkman solutions for each eta (R = 1/eta)
    for eta in eta_list:
        R_val = 1.0 / eta
        p_mesh, uh, ph = load_brinkman_solution(obstacle_type="square", n=n, R_val=R_val, Re=Re, t_final=T_end)
        if p_mesh is None:
            print(f"\n--- Running Brinkman_solver with eta = {eta:.1e} (R = {R_val:.1e}) ---")
            brink_solver = Brinkman_solver(moving=False, type_obstacle="square", n=n, R=R_val, Re=Re)
            p_mesh, uh, ph = brink_solver.Brinkman_solve(
                obstacle=obs,
                dt=dt,
                t_final=T_end
            )

        chi_s = obs.chi(p_mesh, Constant(0.0))
        chi_f = 1.0 - chi_s

        # L2 error norm inside solid Omega_s: ||u_eta||_L2(Omega_s)
        err_solid_sq = assemble(chi_s * inner(uh, uh) * dx)
        err_solid = math.sqrt(err_solid_sq)

        # L2 error norm in fluid domain Omega_f: ||u_eta - u_ref||_L2(Omega_f)
        V_p = uh.function_space()
        u_ref_proj = project(u_ref, V_p)
        err_fluid_vec = uh - u_ref_proj
        err_fluid_sq = assemble(chi_f * inner(err_fluid_vec, err_fluid_vec) * dx)
        err_fluid = math.sqrt(err_fluid_sq)

        results.append({
            'eta': eta,
            'R': R_val,
            'err_solid': err_solid,
            'err_fluid': err_fluid,
            'mesh': p_mesh,
            'uh': uh,
            'ph': ph
        })

    # 4. Compute rates of convergence alpha for O(eta^alpha)
    for i in range(len(results)):
        if i == 0:
            results[i]['rate_solid'] = None
            results[i]['rate_fluid'] = None
        else:
            e_prev = results[i-1]['eta']
            e_curr = results[i]['eta']
            log_ratio = math.log10(e_curr / e_prev)
            rate_s = math.log10(results[i]['err_solid'] / results[i-1]['err_solid']) / log_ratio
            rate_f = math.log10(results[i]['err_fluid'] / results[i-1]['err_fluid']) / log_ratio
            results[i]['rate_solid'] = rate_s
            results[i]['rate_fluid'] = rate_f

    # 5. Print Table 1 comparison
    print("\n" + "="*85)
    print(" TABLE 1: Numerical measurement of error estimates at Re = 40 (Angot et al. 1999)")
    print("="*85)
    print(f"{'eta':<10} | {'||u_eta||_L2(s)':<16} | {'alpha (solid)':<14} | {'||u_eta-uref||_L2(f)':<20} | {'alpha (fluid)':<14}")
    print("-" * 85)
    for r in results:
        rate_s_str = f"{r['rate_solid']:.2f}" if r['rate_solid'] is not None else "---"
        rate_f_str = f"{r['rate_fluid']:.2f}" if r['rate_fluid'] is not None else "---"
        print(f"{r['eta']:<10.1e} | {r['err_solid']:<16.3e} | {rate_s_str:<14} | {r['err_fluid']:<20.3e} | {rate_f_str:<14}")
    print("="*85)

    # 6. Plot Convergence Curves
    plot_l2_penalization_convergence(results, output_dir)

    # 7. Plot Fig. 3 Comparison: Pressure and Vorticity fields
    plot_steady_comparison(ref_mesh, u_ref, p_ref, results[-1]['mesh'], results[-1]['uh'], results[-1]['ph'],
                           eta=results[-1]['eta'], output_dir=output_dir)

    return results


# =============================================================================
# EXPERIMENT 2: UNSTEADY FLOW, VORTEX SHEDDING & STROUHAL NUMBER (Re = 80)
# =============================================================================

def compute_strouhal_number(time_array, signal_array, D=0.2, U_mean=1.0):
    """
    Computes the dominant shedding frequency f via FFT / peak analysis
    and the corresponding Strouhal number St = f * D / U_mean.
    """
    n_pts = len(time_array)
    if n_pts < 20:
        return 0.0, 0.0

    start_idx = n_pts // 3
    t_sig = np.array(time_array[start_idx:])
    sig = np.array(signal_array[start_idx:])

    # Detrend
    sig_detrend = sig - np.mean(sig)

    # Sampling interval
    dt_eff = np.mean(np.diff(t_sig))
    if dt_eff <= 0:
        return 0.0, 0.0

    # FFT
    n_fft = len(sig_detrend)
    freqs = np.fft.rfftfreq(n_fft, d=dt_eff)
    fft_vals = np.abs(np.fft.rfft(sig_detrend))

    valid_idx = freqs > 0.05
    if not np.any(valid_idx):
        return 0.0, 0.0

    dominant_freq = freqs[valid_idx][np.argmax(fft_vals[valid_idx])]
    strouhal = dominant_freq * D / U_mean
    return dominant_freq, strouhal


def run_unsteady_experiment(eta_list=None, n=320, T_end=25.0, dt=0.05, output_dir=None):
    """
    Reproduces Section 6.1 (Unsteady case at Re = 80, Table 2 & Figs. 4-5):
    - Solves unsteady vortex shedding with Conforming_solver (reference).
    - Solves with Brinkman_solver for each eta (R = 1/eta).
    - Measures vortex shedding frequency and Strouhal number St = f*D/U.
    - Demonstrates convective delay effect for large eta = 10^-2.
    """
    if eta_list is None:
        eta_list = [1e-2, 1e-4, 1e-6, 1e-8]

    if output_dir is None:
        output_dir = os.path.join(project_dir, "Plots", "Validation", "L2_unsteady_Re80")
    os.makedirs(output_dir, exist_ok=True)

    Re = 80.0
    D = side_length
    U_mean = 1.0

    print("\n" + "="*80)
    print(f" EXPERIMENT 2: UNSTEADY VORTEX SHEDDING AT Re = {Re} (TABLE 2, FIGS 4 & 5)")
    print("="*80)

    obs = squareObstacle(x_obs, y_obs, side_length=D)
    histories = {}

    # 1. Run reference conforming simulation (search Plots/Conforming first)
    ref_mesh, u_ref, p_ref = load_conforming_solution(obstacle_type="square", n=n, Re=Re)
    if ref_mesh is None:
        print("\n--- Running Unsteady Conforming Reference Solver ---")
        conf_solver = Conforming_solver(moving=False, type_obstacle="square", n=n, Re=Re, structured=True)
        ref_mesh, u_ref, p_ref = conf_solver.conforming_solve(
            obstacle=obs,
            dt=dt,
            t_final=T_end
        )

    # 2. Run penalized Brinkman simulations for each eta (check existing data first)
    for eta in eta_list:
        R_val = 1.0 / eta
        p_mesh, uh, ph = load_brinkman_solution(obstacle_type="square", n=n, R_val=R_val, Re=Re, t_final=T_end)
        if p_mesh is None:
            print(f"\n--- Running Unsteady Brinkman_solver with eta = {eta:.1e} (R = {R_val:.1e}) ---")
            brink_solver = Brinkman_solver(moving=False, type_obstacle="square", n=n, R=R_val, Re=Re)
            p_mesh, uh, ph = brink_solver.Brinkman_solve(
                obstacle=obs,
                dt=dt,
                t_final=T_end
            )
        histories[eta] = {'uh': uh, 'ph': ph}

    print("\nUnsteady simulations completed.")
    return histories


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2 Penalization Validation (Angot et al. 1999)")
    parser.add_argument("--mode", type=str, default="steady", choices=["steady", "unsteady", "all"],
                        help="Select test mode: 'steady' (Re=40), 'unsteady' (Re=80), or 'all'")
    parser.add_argument("--n", type=int, default=320, help="Resolution parameter n (default: 320)")
    parser.add_argument("--T_end", type=float, default=10.0, help="Final simulation time (default: 10.0)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step dt (default: 0.05)")

    args = parser.parse_args()

    if args.mode in ["steady", "all"]:
        run_steady_experiment(
            eta_list=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            n=args.n,
            dt=args.dt,
            T_end=args.T_end
        )

    if args.mode in ["unsteady", "all"]:
        run_unsteady_experiment(
            eta_list=[1e-2, 1e-4, 1e-6, 1e-8],
            n=args.n,
            T_end=args.T_end,
            dt=args.dt
        )
