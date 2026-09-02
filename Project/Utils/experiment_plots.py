"""
Module for generating and saving experimental plots and printing summary tables
for NAPDE project experiments.

Includes:
- Convergence rate computation
- Summary tables printing (Phase comparison, Spatial convergence, Strategy A, Strategy B)
- Phase comparison convergence plots (Conforming vs Buffer method)
- Spatial grid convergence plots with interface trace error
- Brinkman Strategy A (Penalty R-sweep at fixed mesh size)
- Brinkman Strategy B (Balanced scaling R(h) ~ h^-2)
- Vertical interface velocity profile recovery plots
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONVERGENCE RATE COMPUTATION & SUMMARY TABLES
# =============================================================================

def compute_convergence_rates(err_list: List[float], h_list: List[float]) -> List[float]:
    """
    Computes empirical convergence rates between consecutive mesh resolutions:
    Rate = ln(E_i / E_{i+1}) / ln(h_i / h_{i+1})
    """
    return [np.log(err_list[i] / err_list[i+1]) / np.log(h_list[i] / h_list[i+1]) for i in range(len(h_list) - 1)]


def print_phase_comparison_tables(
    resolutions: List[int],
    h_vals: List[float],
    res_p1: Dict[str, List[float]],
    res_p2: Dict[str, List[float]],
    method_name: str
):
    """
    Prints formatted convergence tables comparing Phase 1 (Conforming) and Phase 2 (Buffer recovery),
    including L2(u), H1(u), and L2(p) convergence rates.
    """
    rates_p1_L2 = compute_convergence_rates(res_p1["L2_u"], h_vals)
    rates_p1_H1 = compute_convergence_rates(res_p1["H1_u"], h_vals)
    rates_p1_p  = compute_convergence_rates(res_p1["L2_p"], h_vals) if "L2_p" in res_p1 else []

    rates_p2_L2 = compute_convergence_rates(res_p2["L2_u"], h_vals)
    rates_p2_H1 = compute_convergence_rates(res_p2["H1_u"], h_vals)
    rates_p2_p  = compute_convergence_rates(res_p2["L2_p"], h_vals) if "L2_p" in res_p2 else []

    print("\n" + "=" * 98)
    print("Table 1: Phase 1 - BENCHMARK CONFORMING (Omega_0)")
    print("=" * 98)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'L2(p) Error':>14} | {'Rate':>6}")
    print("-" * 98)
    for i, n in enumerate(resolutions):
        r_l2 = f"{rates_p1_L2[i-1]:+5.2f}" if i > 0 else "   -- "
        r_h1 = f"{rates_p1_H1[i-1]:+5.2f}" if i > 0 else "   -- "
        r_p  = f"{rates_p1_p[i-1]:+5.2f}" if (i > 0 and len(rates_p1_p) > 0) else "   -- "
        p_val_str = f"{res_p1['L2_p'][i]:14.5e}" if "L2_p" in res_p1 else "           N/A"
        print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p1['L2_u'][i]:14.5e} | {r_l2} | {res_p1['H1_u'][i]:14.5e} | {r_h1} | {p_val_str} | {r_p}")

    has_p2_p = "L2_p" in res_p2 and len(res_p2["L2_p"]) == len(resolutions)
    if has_p2_p:
        print("\n" + "=" * 118)
        print(f"Table 2: Phase 2 - UPSTREAM BUFFER RECOVERY VIA {method_name.upper()} (Restricted to Omega_0)")
        print("=" * 118)
        print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'L2(p) Error':>14} | {'Rate':>6} | {'Interface L2':>14}")
        print("-" * 118)
        for i, n in enumerate(resolutions):
            r_l2 = f"{rates_p2_L2[i-1]:+5.2f}" if i > 0 else "   -- "
            r_h1 = f"{rates_p2_H1[i-1]:+5.2f}" if i > 0 else "   -- "
            r_p  = f"{rates_p2_p[i-1]:+5.2f}" if (i > 0 and len(rates_p2_p) > 0) else "   -- "
            print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p2['L2_u'][i]:14.5e} | {r_l2} | {res_p2['H1_u'][i]:14.5e} | {r_h1} | {res_p2['L2_p'][i]:14.5e} | {r_p} | {res_p2['interf_L2'][i]:14.5e}")
        print("=" * 118 + "\n")
    else:
        print("\n" + "=" * 98)
        print(f"Table 2: Phase 2 - UPSTREAM BUFFER RECOVERY VIA {method_name.upper()} (Restricted to Omega_0)")
        print("=" * 98)
        print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'Interface L2':>14}")
        print("-" * 98)
        for i, n in enumerate(resolutions):
            r_l2 = f"{rates_p2_L2[i-1]:+5.2f}" if i > 0 else "   -- "
            r_h1 = f"{rates_p2_H1[i-1]:+5.2f}" if i > 0 else "   -- "
            print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p2['L2_u'][i]:14.5e} | {r_l2} | {res_p2['H1_u'][i]:14.5e} | {r_h1} | {res_p2['interf_L2'][i]:14.5e}")
        print("=" * 98 + "\n")


def print_spatial_convergence_table(
    resolutions: List[int],
    h_vals: List[float],
    errs_L2_u: List[float],
    errs_H1_u: List[float],
    errs_intf: List[float],
    method_name: str,
    errs_L2_p: List[float] = None
):
    """
    Prints a single method convergence summary table with rates for L2(u), H1(u), L2(p), and Interface trace.
    """
    rates_L2 = compute_convergence_rates(errs_L2_u, h_vals)
    rates_H1 = compute_convergence_rates(errs_H1_u, h_vals)
    rates_intf = compute_convergence_rates(errs_intf, h_vals)
    rates_p = compute_convergence_rates(errs_L2_p, h_vals) if errs_L2_p is not None else None

    if rates_p is not None:
        print("\n" + "=" * 128)
        print(f"CONVERGENCE SUMMARY TABLE: {method_name.upper()} BUFFER RECOVERY")
        print("=" * 128)
        print(f"{'n':>5} | {'h':>8} | {'L2(u) Err (Omega_0)':>20} | {'Rate':>6} | {'H1(u) Err':>12} | {'Rate':>6} | {'L2(p) Err':>12} | {'Rate':>6} | {'Interface L2':>14} | {'Rate':>6}")
        print("-" * 128)

        for i, n in enumerate(resolutions):
            r_l2_str = f"{rates_L2[i-1]:+6.2f}" if i > 0 else "    --"
            r_h1_str = f"{rates_H1[i-1]:+6.2f}" if i > 0 else "    --"
            r_p_str = f"{rates_p[i-1]:+6.2f}" if i > 0 else "    --"
            r_intf_str = f"{rates_intf[i-1]:+6.2f}" if i > 0 else "    --"
            print(f"{n:5d} | {h_vals[i]:8.4f} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_H1_u[i]:12.5e} | {r_h1_str} | {errs_L2_p[i]:12.5e} | {r_p_str} | {errs_intf[i]:14.5e} | {r_intf_str}")
        print("=" * 128 + "\n")
    else:
        print("\n" + "=" * 105)
        print(f"CONVERGENCE SUMMARY TABLE: {method_name.upper()} BUFFER RECOVERY")
        print("=" * 105)
        print(f"{'n':>5} | {'h':>8} | {'L2(u) Err (Omega_0)':>20} | {'Rate':>6} | {'H1(u) Err':>12} | {'Rate':>6} | {'Interface L2':>14} | {'Rate':>6}")
        print("-" * 105)

        for i, n in enumerate(resolutions):
            r_l2_str = f"{rates_L2[i-1]:+6.2f}" if i > 0 else "    --"
            r_h1_str = f"{rates_H1[i-1]:+6.2f}" if i > 0 else "    --"
            r_intf_str = f"{rates_intf[i-1]:+6.2f}" if i > 0 else "    --"
            print(f"{n:5d} | {h_vals[i]:8.4f} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_H1_u[i]:12.5e} | {r_h1_str} | {errs_intf[i]:14.5e} | {r_intf_str}")
        print("=" * 105 + "\n")


def print_strategy_a_table(
    R_values: List[float],
    errs_L2_u: List[float],
    errs_intf: List[float]
):
    """
    Prints Strategy A summary table: Error as a function of Brinkman penalty R.
    """
    rates_L2 = [np.log(errs_L2_u[i] / errs_L2_u[i+1]) / np.log(R_values[i+1] / R_values[i]) for i in range(len(R_values)-1)]
    rates_intf = [np.log(errs_intf[i] / errs_intf[i+1]) / np.log(R_values[i+1] / R_values[i]) for i in range(len(R_values)-1)]

    print("\n" + "=" * 95)
    print("CONVERGENCE SUMMARY TABLE: ERROR AS A FUNCTION OF BRINKMAN PENALTY (R)")
    print("=" * 95)
    print(f"{'R Penalty':>12} | {'L2(u) Err (Omega_0)':>20} | {'Rate(1/R)':>10} | {'Interface L2 Err':>18} | {'Rate(1/R)':>10}")
    print("-" * 95)
    for i, R_val in enumerate(R_values):
        r_l2_str = f"{rates_L2[i-1]:+7.3f}" if i > 0 else "      --"
        r_intf_str = f"{rates_intf[i-1]:+7.3f}" if i > 0 else "      --"
        print(f"{R_val:12.1e} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_intf[i]:18.5e} | {r_intf_str}")
    print("=" * 95 + "\n")


def print_strategy_b_table(
    resolutions: List[int],
    h_vals: List[float],
    scaled_R_vals: List[float],
    errs_L2_u: List[float],
    errs_H1_u: List[float],
    errs_intf: List[float],
    errs_L2_p: List[float] = None
):
    """
    Prints Strategy B summary table: Spatial convergence with balanced penalty scaling R(h),
    including L2(u), H1(u), and L2(p) convergence rates.
    """
    rates_L2 = compute_convergence_rates(errs_L2_u, h_vals)
    rates_H1 = compute_convergence_rates(errs_H1_u, h_vals)
    rates_p  = compute_convergence_rates(errs_L2_p, h_vals) if errs_L2_p is not None else None

    if rates_p is not None:
        print("\n" + "=" * 128)
        print("CONVERGENCE SUMMARY TABLE: SPATIAL CONVERGENCE WITH SCALED BRINKMAN PENALTY R(h)")
        print("=" * 128)
        print(f"{'n':>5} | {'h':>8} | {'Scaled R':>12} | {'L2(u) Err (Omega_0)':>20} | {'Rate':>6} | {'H1(u) Err':>12} | {'Rate':>6} | {'L2(p) Err':>12} | {'Rate':>6} | {'Interface L2':>14}")
        print("-" * 128)
        for i, n in enumerate(resolutions):
            r_l2_str = f"{rates_L2[i-1]:+6.2f}" if i > 0 else "    --"
            r_h1_str = f"{rates_H1[i-1]:+6.2f}" if i > 0 else "    --"
            r_p_str  = f"{rates_p[i-1]:+6.2f}" if i > 0 else "    --"
            print(f"{n:5d} | {h_vals[i]:8.4f} | {scaled_R_vals[i]:12.2e} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_H1_u[i]:12.5e} | {r_h1_str} | {errs_L2_p[i]:12.5e} | {r_p_str} | {errs_intf[i]:14.5e}")
        print("=" * 128 + "\n")
    else:
        print("\n" + "=" * 105)
        print("CONVERGENCE SUMMARY TABLE: SPATIAL CONVERGENCE WITH SCALED BRINKMAN PENALTY R(h)")
        print("=" * 105)
        print(f"{'n':>5} | {'h':>8} | {'Scaled R':>12} | {'L2(u) Err (Omega_0)':>20} | {'Rate':>6} | {'H1(u) Err':>12} | {'Rate':>6} | {'Interface L2':>14}")
        print("-" * 105)
        for i, n in enumerate(resolutions):
            r_l2_str = f"{rates_L2[i-1]:+6.2f}" if i > 0 else "    --"
            r_h1_str = f"{rates_H1[i-1]:+6.2f}" if i > 0 else "    --"
            print(f"{n:5d} | {h_vals[i]:8.4f} | {scaled_R_vals[i]:12.2e} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_H1_u[i]:12.5e} | {r_h1_str} | {errs_intf[i]:14.5e}")
        print("=" * 105 + "\n")


# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def plot_phase_comparison_loglog(
    h_vals: List[float],
    res_p1: Dict[str, List[float]],
    res_p2: Dict[str, List[float]],
    method_name: str,
    output_path: str
):
    """
    Plots log-log spatial convergence comparing Phase 1 (Conforming) and Phase 2 (Buffer recovery).
    """
    h_arr = np.array(h_vals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Spatial Convergence: Pure Conforming vs {method_name} Buffer Recovery", fontsize=13, fontweight='bold')

    # Subplot 1: L2 Velocity
    ax1.loglog(h_arr, res_p1["L2_u"], 'o-', color='#1f77b4', linewidth=2, label=r'Phase 1: Conforming $L^2(\mathbf{u})$')
    ax1.loglog(h_arr, res_p2["L2_u"], 's--', color='#d62728', linewidth=2, label=f'Phase 2: {method_name} $L^2(\\mathbf{{u}})$')
    ax1.loglog(h_arr, [res_p1["L2_u"][0] * (h / h_vals[0])**2 for h in h_arr], 'k:', alpha=0.6, label=r'Optimal $O(h^2)$')
    ax1.loglog(h_arr, [res_p1["L2_u"][0] * (h / h_vals[0])**1 for h in h_arr], 'k--', alpha=0.6, label=r'Slope $O(h)$')
    ax1.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax1.set_ylabel(r"Velocity $L^2$ Error in $\Omega_0$", fontsize=11)
    ax1.set_title(r"Velocity $L^2$ Error Norm", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    # Subplot 2: H1 Velocity
    ax2.loglog(h_arr, res_p1["H1_u"], 'o-', color='#1f77b4', linewidth=2, label=r'Phase 1: Conforming $H^1(\mathbf{u})$')
    ax2.loglog(h_arr, res_p2["H1_u"], 's--', color='#d62728', linewidth=2, label=f'Phase 2: {method_name} $H^1(\\mathbf{{u}})$')
    ax2.loglog(h_arr, [res_p1["H1_u"][0] * (h / h_vals[0])**1 for h in h_arr], 'k--', alpha=0.6, label=r'Optimal $O(h)$')
    ax2.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax2.set_ylabel(r"Velocity $H^1$ Error in $\Omega_0$", fontsize=11)
    ax2.set_title(r"Velocity $H^1$ Error Norm", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Log-Log Convergence Plot: {output_path}")


def plot_spatial_convergence_with_interface(
    h_vals: List[float],
    errs_L2_u: List[float],
    errs_H1_u: List[float],
    errs_intf: List[float],
    method_name: str,
    output_path: str
):
    """
    Plots spatial convergence for a single buffer method (L2, H1 and interface trace error).
    """
    h_arr = np.array(h_vals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Spatial Convergence: Upstream Buffer Recovery via {method_name}", fontsize=13, fontweight='bold')

    # Subplot 1: Domain errors
    ax1.loglog(h_arr, errs_L2_u, 'o-', color='#1f77b4', linewidth=2, label=r'Velocity $L^2(\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_H1_u, 's--', color='#ff7f0e', linewidth=1.8, label=r'Velocity $H^1(\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label=r'Optimal $O(h^2)$')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label=r'Slope $O(h)$')
    ax1.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax1.set_ylabel(r"Velocity Error Norms in $\Omega_0$", fontsize=11)
    ax1.set_title("Restricted Velocity Error in Physical Domain", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    # Subplot 2: Interface trace error
    ax2.loglog(h_arr, errs_intf, 'd-', color='#d62728', linewidth=2, label=r'Interface Trace $L^2(\Sigma)$ Error')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label=r'Optimal $O(h^2)$')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label=r'Slope $O(h)$')
    ax2.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax2.set_ylabel(r"Trace Error at Interface $\Sigma$ ($x = 0$)", fontsize=11)
    ax2.set_title(r"Dirichlet Interface Recovery vs $h$", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Convergence Plot: {output_path}")


def plot_strategy_a_r_sweep(
    R_values: List[float],
    errs_L2_u: List[float],
    errs_H1_u: List[float],
    errs_L2_p: List[float],
    errs_intf: List[float],
    fixed_n: int,
    output_path: str
):
    """
    Plots Strategy A: Asymptotic modeling error vs Brinkman penalty R at fixed mesh size.
    """
    R_arr = np.array(R_values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Strategy A: Asymptotic Modeling Error vs Penalty $R$ ($n = {fixed_n}$)", fontsize=13, fontweight='bold')

    ax1.loglog(R_arr, errs_L2_u, 'o-', color='#1f77b4', linewidth=2, label=r'Velocity $L^2(\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_H1_u, 's--', color='#ff7f0e', linewidth=1.8, label=r'Velocity $H^1(\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_L2_p, '^:', color='#2ca02c', linewidth=1.8, label=r'Pressure $L^2(\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_L2_u[0] * (R_arr[0] / R_arr)**0.5, 'k--', alpha=0.5, label=r'Asymptotic $O(R^{-1/2})$')
    ax1.loglog(R_arr, errs_L2_u[0] * (R_arr[0] / R_arr)**1.0, 'k:', alpha=0.5, label=r'Asymptotic $O(R^{-1})$')
    ax1.set_xlabel(r"Brinkman Penalty Parameter $R$", fontsize=11)
    ax1.set_ylabel(r"Error Norms in Physical Domain $\Omega_0$", fontsize=11)
    ax1.set_title(r"Modeling Error in $\Omega_0$ vs $R$", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.loglog(R_arr, errs_intf, 'd-', color='#d62728', linewidth=2, label=r'Interface Trace $L^2(\Sigma)$ Error')
    ax2.loglog(R_arr, errs_intf[0] * (R_arr[0] / R_arr)**0.5, 'k--', alpha=0.5, label=r'Asymptotic $O(R^{-1/2})$')
    ax2.loglog(R_arr, errs_intf[0] * (R_arr[0] / R_arr)**1.0, 'k:', alpha=0.5, label=r'Asymptotic $O(R^{-1})$')
    ax2.set_xlabel(r"Brinkman Penalty Parameter $R$", fontsize=11)
    ax2.set_ylabel(r"Trace Error at Interface $\Sigma$ ($x = 0$)", fontsize=11)
    ax2.set_title(r"Dirichlet Recovery Error vs $R$", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Strategy A Error Plot: {output_path}")


def plot_strategy_b_spatial_convergence(
    h_vals: List[float],
    errs_L2_u: List[float],
    errs_H1_u: List[float],
    errs_intf: List[float],
    output_path: str
):
    """
    Plots Strategy B: Spatial convergence with balanced penalty scaling R(h) ~ h^-2.
    """
    h_arr = np.array(h_vals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(r"Strategy B: Spatial Convergence with Scaled Penalty $R(h) = R_0 \cdot (h_0/h)^2$", fontsize=13, fontweight='bold')

    ax1.loglog(h_arr, errs_L2_u, 'o-', color='#1f77b4', linewidth=2, label=r'Velocity $L^2(\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_H1_u, 's--', color='#ff7f0e', linewidth=1.8, label=r'Velocity $H^1(\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label=r'Reference $O(h^2)$')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label=r'Reference $O(h)$')
    ax1.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax1.set_ylabel(r"Velocity Error Norms in $\Omega_0$", fontsize=11)
    ax1.set_title(r"Restricted Velocity Error vs $h$", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    ax2.loglog(h_arr, errs_intf, 'd-', color='#d62728', linewidth=2, label=r'Interface Trace $L^2(\Sigma)$ Error')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label=r'Reference $O(h^2)$')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label=r'Reference $O(h)$')
    ax2.set_xlabel(r"Mesh Size $h = L_x/n$", fontsize=11)
    ax2.set_ylabel(r"Trace Error at Interface $\Sigma$ ($x = 0$)", fontsize=11)
    ax2.set_title(r"Dirichlet Interface Recovery vs $h$", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Strategy B Convergence Plot: {output_path}")


def plot_interface_velocity_profile(
    profiles: Dict[Union[int, float], Tuple[np.ndarray, np.ndarray]],
    Ly: float,
    keys: List[Union[int, float]],
    title: str,
    output_path: str,
    custom_labels: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None
):
    """
    Plots the horizontal velocity cut u_x(0, y) along the interface x = 0 compared to the analytical value 1.0.
    """
    fig_prof, ax_prof = plt.subplots(figsize=(7, 6))
    fig_prof.suptitle(title, fontsize=12, fontweight='bold')

    y_fine = np.linspace(0.0, Ly, 200)
    u_ex_line = np.ones_like(y_fine)  # Target exact u_x(0, y) = 1.0
    ax_prof.plot(u_ex_line, y_fine, 'k-', linewidth=2.5, label=r'Target $\mathbf{u}_{ex}(0, y) = 1.0$')

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(keys)))
    for idx, k in enumerate(keys):
        y_p, u_p = profiles[k]
        lbl = custom_labels[idx] if custom_labels is not None else f'n = {k}'
        ax_prof.plot(u_p, y_p, '--', color=colors[idx], linewidth=1.8, label=lbl)

    ax_prof.set_xlabel(r"Horizontal Velocity $u_x(0, y)$", fontsize=11)
    ax_prof.set_ylabel(r"Channel Height $y$", fontsize=11)
    if xlim is not None:
        ax_prof.set_xlim(xlim)
    ax_prof.set_ylim(0.0, Ly)
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig_prof)
    print(f">> Saved Interface Profile Plot: {output_path}")
