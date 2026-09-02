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
    load_conforming_solution, load_brinkman_solution, extract_probe_history
)
from validation.validation_plots import (
    plot_l2_penalization_convergence,
    plot_pressure_comparison,
    plot_vorticity_comparison,
    plot_strouhal_comparison
)


# =============================================================================
# EXPERIMENT 1: STEADY STATE CONVERGENCE & TABLE 1 (Re = 40)
# =============================================================================

def run_steady_experiment(eta_list=None, n=320, T_end=40.0, dt=0.2, structured=True, output_dir=None):
    """
    Reproduces Section 6.1 (Steady case at Re = 40, Table 1 & Fig. 3):
    - Solves conforming reference (Dirichlet on square) using Conforming_solver.
    - Solves penalized solutions using Brinkman_solver for each eta in eta_list (R = 1/eta).
    - Computes L2 errors in solid Omega_s and fluid Omega_f and spatial convergence rates.
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
    ref_mesh, u_ref, p_ref = load_conforming_solution(obstacle_type="square", n=n, Re=Re, t_final=T_end)
    if ref_mesh is None:
        print("\n--- Running Conforming Reference Solver ---")
        conf_solver = Conforming_solver(moving=False, type_obstacle="square", n=n, Re=Re, structured=structured)
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
            brink_solver = Brinkman_solver(moving=False, type_obstacle="square", n=n, R=R_val, Re=Re, structured=structured)
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

        # L2 error norm of pressure in fluid domain Omega_f: ||p_eta - p_ref||_L2(Omega_f)
        Q_p = ph.function_space()
        p_ref_proj = project(p_ref, Q_p)
        vol_fluid = assemble(chi_f * dx)
        mean_pref = assemble(chi_f * p_ref_proj * dx) / vol_fluid
        mean_ph   = assemble(chi_f * ph * dx) / vol_fluid
        err_p_vec = (ph - mean_ph) - (p_ref_proj - mean_pref)
        err_p_sq  = assemble(chi_f * inner(err_p_vec, err_p_vec) * dx)
        err_pressure = math.sqrt(err_p_sq)

        results.append({
            'eta': eta,
            'R': R_val,
            'err_solid': err_solid,
            'err_fluid': err_fluid,
            'err_pressure': err_pressure,
            'mesh': p_mesh,
            'uh': uh,
            'ph': ph
        })

    # 4. Compute rates of convergence alpha for O(eta^alpha)
    for i in range(len(results)):
        if i == 0:
            results[i]['rate_solid'] = None
            results[i]['rate_fluid'] = None
            results[i]['rate_pressure'] = None
        else:
            e_prev = results[i-1]['eta']
            e_curr = results[i]['eta']
            log_ratio = math.log10(e_curr / e_prev)
            rate_s = math.log10(results[i]['err_solid'] / results[i-1]['err_solid']) / log_ratio
            rate_f = math.log10(results[i]['err_fluid'] / results[i-1]['err_fluid']) / log_ratio
            rate_p = math.log10(results[i]['err_pressure'] / results[i-1]['err_pressure']) / log_ratio
            results[i]['rate_solid'] = rate_s
            results[i]['rate_fluid'] = rate_f
            results[i]['rate_pressure'] = rate_p

    # 5. Print Table 1 comparison
    print("\n" + "="*118)
    print(" TABLE 1: Numerical measurement of error estimates at Re = 40 (Angot et al. 1999)")
    print("="*118)
    print(f"{'eta':<10} | {'||u_eta||_L2(s)':<16} | {'alpha (solid)':<14} | {'||u-uref||_L2(f)':<18} | {'alpha (fluid)':<14} | {'||p-pref||_L2(f)':<18} | {'alpha (p)':<10}")
    print("-" * 118)
    for r in results:
        rate_s_str = f"{r['rate_solid']:.2f}" if r['rate_solid'] is not None else "---"
        rate_f_str = f"{r['rate_fluid']:.2f}" if r['rate_fluid'] is not None else "---"
        rate_p_str = f"{r['rate_pressure']:.2f}" if r['rate_pressure'] is not None else "---"
        print(f"{r['eta']:<10.1e} | {r['err_solid']:<16.3e} | {rate_s_str:<14} | {r['err_fluid']:<18.3e} | {rate_f_str:<14} | {r['err_pressure']:<18.3e} | {rate_p_str:<10}")
    print("="*118)

    # 6. Plot 1: Log-log Convergence Curves (Table 1)
    plot_l2_penalization_convergence(results, output_dir)

    # 7. Plot 2: Pressure Comparison (Colormap + Isobars)
    plot_pressure_comparison(ref_mesh, p_ref, results[-1]['mesh'], results[-1]['ph'],
                             eta=results[-1]['eta'], output_dir=output_dir,
                             x_obs=x_obs, y_obs=y_obs, side_length=D)

    # 8. Plot 3: Vorticity Comparison (Colormap + Isolines)
    plot_vorticity_comparison(ref_mesh, u_ref, results[-1]['mesh'], results[-1]['uh'],
                              eta=results[-1]['eta'], output_dir=output_dir,
                              x_obs=x_obs, y_obs=y_obs, side_length=D)

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


def run_unsteady_experiment(eta_list=None, n=320, T_end=25.0, dt=0.05, structured=True, output_dir=None):
    """
    Reproduces Section 6.1 (Unsteady case at Re = 80, Table 2 & Figs. 4-5):
    - Solves unsteady vortex shedding with Conforming_solver (reference).
    - Solves with Brinkman_solver for each eta (R = 1/eta).
    - Computes shedding frequency f, period T, and Strouhal number St = f*D/U.
    - Generates 3 validation plots: Strouhal & Signal comparison, Pressure, and Vorticity street (von Kármán).
    """
    if eta_list is None:
        eta_list = [1e-2, 1e-4, 1e-6, 1e-8]

    if output_dir is None:
        output_dir = os.path.join(project_dir, "Plots", "Validation", "L2_unsteady_Re80")
    os.makedirs(output_dir, exist_ok=True)

    Re = 40
    D = side_length
    U_mean = 1.0
    probe_pt = (1.5, 0.5)

    print("\n" + "="*80)
    print(f" EXPERIMENT 2: UNSTEADY VORTEX SHEDDING AT Re = {Re} (TABLE 2)")
    print("="*80)

    obs = squareObstacle(x_obs, y_obs, side_length=D)

    # 1. Run reference conforming simulation (search Plots/Conforming first)
    ref_mesh, u_ref, p_ref = load_conforming_solution(obstacle_type="square", n=n, Re=Re, t_final=T_end)
    if ref_mesh is None:
        print("\n--- Running Unsteady Conforming Reference Solver ---")
        conf_solver = Conforming_solver(moving=False, type_obstacle="square", n=n, Re=Re, structured=structured)
        ref_mesh, u_ref, p_ref = conf_solver.conforming_solve(
            obstacle=obs,
            dt=dt,
            t_final=T_end
        )

    # Extract Conforming Reference probe signal & Strouhal number
    conf_dir = get_case_directory("Conforming", "square", Re=Re, n=n)
    t_ref, ux_ref, uy_ref = extract_probe_history(conf_dir, probe_pt=probe_pt)
    f_ref, St_ref = compute_strouhal_number(t_ref, uy_ref, D=D, U_mean=U_mean)
    T_ref = (1.0 / f_ref) if f_ref > 0 else 0.0

    strouhal_data = {
        'Conforming': {
            'label': 'Conforming (Dirichlet)',
            't': t_ref,
            'uy': uy_ref,
            'f': f_ref,
            'T': T_ref,
            'St': St_ref
        }
    }

    results = []

    # 2. Run penalized Brinkman simulations for each eta (check existing data first)
    for eta in eta_list:
        R_val = 1.0 / eta
        p_mesh, uh, ph = load_brinkman_solution(obstacle_type="square", n=n, R_val=R_val, Re=Re, t_final=T_end)
        if p_mesh is None:
            print(f"\n--- Running Unsteady Brinkman_solver with eta = {eta:.1e} (R = {R_val:.1e}) ---")
            brink_solver = Brinkman_solver(moving=False, type_obstacle="square", n=n, R=R_val, Re=Re, structured=structured)
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

        # Extract Brinkman probe signal & Strouhal number
        brink_dir = get_case_directory("Brinkman", "square", Re=Re, n=n, R_penalty=R_val)
        t_pen, ux_pen, uy_pen = extract_probe_history(brink_dir, probe_pt=probe_pt)
        f_pen, St_pen = compute_strouhal_number(t_pen, uy_pen, D=D, U_mean=U_mean)
        T_pen = (1.0 / f_pen) if f_pen > 0 else 0.0

        strouhal_data[eta] = {
            'label': fr'$\eta = {eta:.1e}$',
            't': t_pen,
            'uy': uy_pen,
            'f': f_pen,
            'T': T_pen,
            'St': St_pen
        }

        results.append({
            'eta': eta,
            'R': R_val,
            'err_solid': err_solid,
            'err_fluid': err_fluid,
            'mesh': p_mesh,
            'uh': uh,
            'ph': ph,
            'St': St_pen,
            'f': f_pen,
            'T': T_pen
        })

    # 3. Print Table 2: Strouhal Number & Shedding Frequency at Re = 80
    print("\n" + "="*95)
    print(" TABLE 2: Unsteady flow at Re = 80: Strouhal number and shedding frequency (Angot et al. 1999)")
    print("="*95)
    print(f"{'Case':<25} | {'Period T (s)':<14} | {'Frequency f (Hz)':<18} | {'Strouhal St':<14} | {'St Error (%)':<12}")
    print("-" * 95)
    print(f"{'Conforming Reference':<25} | {T_ref:<14.3f} | {f_ref:<18.3f} | {St_ref:<14.4f} | {'---':<12}")
    for r in results:
        err_st_str = f"{abs(r['St'] - St_ref)/St_ref * 100:.2f}%" if St_ref > 0 else "---"
        print(f"eta = {r['eta']:<18.1e} | {r['T']:<14.3f} | {r['f']:<18.3f} | {r['St']:<14.4f} | {err_st_str:<12}")
    print("="*95)

    # 4. Generate the 3 Output Plots for Unsteady Flow:
    # 4.1. Plot 1: Strouhal Number and Time Signals Comparison (Table 2 & Fig. 4)
    plot_strouhal_comparison(strouhal_data, output_dir)

    # 4.2. Plot 2: Pressure Comparison Plot (Colormap + Isobars at t = T_end)
    plot_pressure_comparison(ref_mesh, p_ref, results[-1]['mesh'], results[-1]['ph'],
                             eta=results[-1]['eta'], output_dir=output_dir,
                             x_obs=x_obs, y_obs=y_obs, side_length=D)

    # 4.3. Plot 3: Vorticity Comparison Plot (Colormap + Isolines showing von Kármán vortex street at t = T_end)
    plot_vorticity_comparison(ref_mesh, u_ref, results[-1]['mesh'], results[-1]['uh'],
                              eta=results[-1]['eta'], output_dir=output_dir,
                              x_obs=x_obs, y_obs=y_obs, side_length=D)

    print(f"\nUnsteady experiment completed. All 3 plots saved to: {output_dir}")
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # PARAMETRI MODIFICABILI DIRETTAMENTE DA CODICE
    # =========================================================================
    # Modalità di test: "steady" (Re=40), "unsteady" (Re=80), o "all"
    mode = "steady"

    # Risoluzione mesh (es. n=320 per benchmark finale, n=80 per test veloci)
    n = 320

    # Passo temporale dt (es. dt=0.5 o dt=0.05)
    dt = 0.2

    # Tempo finale di simulazione
    T_end_steady = 40.0
    T_end_unsteady =15.0

    # Lista di valori di eta (permeabilità = 1/R) da testare
    eta_list_steady = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    eta_list_unsteady = [1e-2, 1e-4, 1e-6,1e-8]
    # =========================================================================

    parser = argparse.ArgumentParser(description="L2 Penalization Validation (Angot et al. 1999)")
    parser.add_argument("--mode", type=str, default=mode, choices=["steady", "unsteady", "all"],
                        help="Select test mode: 'steady' (Re=40), 'unsteady' (Re=80), or 'all'")
    parser.add_argument("--unstructured", action="store_true", default=False,
                        help="Use unstructured mesh instead of structured")
    args = parser.parse_args()

    use_structured = not args.unstructured

    if args.mode in ["steady", "all"]:
        run_steady_experiment(
            eta_list=eta_list_steady,
            n=n,
            dt=dt,
            T_end=T_end_steady,
            structured=use_structured
        )

    if args.mode in ["unsteady", "all"]:
        run_unsteady_experiment(
            eta_list=eta_list_unsteady,
            n=n,
            T_end=T_end_unsteady,
            dt=dt,
            structured=use_structured
        )

