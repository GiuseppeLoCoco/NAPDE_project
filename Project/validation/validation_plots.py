"""
Plotting utilities for validation and convergence benchmarks.
Centralizes all visualization functions (convergence curves, field comparisons, profiles, and summary tables).
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function
from firedrake.pyplot import tripcolor


# =============================================================================
# 1. L2 PENALIZATION PAPER PLOTS (Angot et al. 1999)
# =============================================================================

def plot_l2_penalization_convergence(results, output_dir: str):
    """
    Plots log-log convergence of L2 velocity error norms in solid and fluid vs penalization parameter eta.
    """
    etas = [r['eta'] for r in results]
    err_s = [r['err_solid'] for r in results]
    err_f = [r['err_fluid'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(etas, err_s, 'o-b', linewidth=2, label=r'$\|u_\eta\|_{L^2(\Omega_s)}$ (Solid error)')
    ax.loglog(etas, err_f, 's-r', linewidth=2, label=r'$\|u_\eta - u_{ref}\|_{L^2(\Omega_f)}$ (Fluid error)')
    ax.loglog(etas, [etas[0]**1.0 * err_s[0] / (etas[0]**1.0) * (e/etas[0]) for e in etas], '--k', label=r'Theoretical $O(\eta)$')
    
    ax.set_xlabel(r'Penalization parameter $\eta$', fontsize=12)
    ax.set_ylabel(r'$L^2$ Error Norm', fontsize=12)
    ax.set_title('Convergence of L2 Penalization Method at Re = 40 (Table 1)', fontsize=13)
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=11)
    
    plot_path_conv = os.path.join(output_dir, "l2_error_convergence_re40.png")
    fig.savefig(plot_path_conv, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved convergence plot to: {plot_path_conv}")


def plot_steady_comparison(ref_mesh, u_ref, p_ref, pen_mesh, u_pen, p_pen, eta: float, output_dir: str):
    """
    Generates side-by-side / stacked comparison of Pressure and Vorticity fields
    between reference Dirichlet condition and L2 penalization (Fig. 3 from paper).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Compute vorticity: curl(u) = du_y/dx - du_x/dy
    V_scalar_ref = FunctionSpace(ref_mesh, "CG", 1)
    vort_ref = Function(V_scalar_ref, name="Vorticity_ref").interpolate(u_ref[1].dx(0) - u_ref[0].dx(1))

    V_scalar_pen = FunctionSpace(pen_mesh, "CG", 1)
    vort_pen = Function(V_scalar_pen, name="Vorticity_pen").interpolate(u_pen[1].dx(0) - u_pen[0].dx(1))

    # 1. Reference Pressure
    axes[0].set_title("Reference (Conforming Dirichlet): Pressure Field p", fontsize=11)
    p0 = tripcolor(p_ref, axes=axes[0], cmap='RdYlBu_r', shading='gouraud')
    fig.colorbar(p0, ax=axes[0], fraction=0.02, pad=0.02)

    # 2. Reference Vorticity
    axes[1].set_title("Reference (Conforming Dirichlet): Vorticity Field ω", fontsize=11)
    p1 = tripcolor(vort_ref, axes=axes[1], cmap='coolwarm', shading='gouraud')
    fig.colorbar(p1, ax=axes[1], fraction=0.02, pad=0.02)

    # 3. L2 Penalization Pressure
    axes[2].set_title(f"L2 Penalization (η = {eta:.1e}): Pressure Field p_η", fontsize=11)
    p2 = tripcolor(p_pen, axes=axes[2], cmap='RdYlBu_r', shading='gouraud')
    fig.colorbar(p2, ax=axes[2], fraction=0.02, pad=0.02)

    # 4. L2 Penalization Vorticity
    axes[3].set_title(f"L2 Penalization (η = {eta:.1e}): Vorticity Field ω_η", fontsize=11)
    p3 = tripcolor(vort_pen, axes=axes[3], cmap='coolwarm', shading='gouraud')
    fig.colorbar(p3, ax=axes[3], fraction=0.02, pad=0.02)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fig3_pressure_vorticity_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Fig. 3 comparison plot to: {plot_path}")


# =============================================================================
# 2. SPATIAL CONVERGENCE STUDY PLOTS (convergence_analysis.py)
# =============================================================================

def plot_spatial_convergence_summary(resolutions, dx_h, err_L2_u, err_H1_u, err_L2_p,
                                     rates_L2_u, rates_H1_u, rates_L2_p,
                                     profiles, y_ref, u_mag_ref,
                                     solver_type: str, obstacle_type: str, Re: float,
                                     x_obs: float, Ly: float, plot_filename: str):
    """
    Generates 3-panel figure with log-log convergence curve, vertical centerline velocity profile,
    and structured error/rate summary tables.
    """
    case_name = "Brinkman" if solver_type.lower() == "brinkman" else "DLM"

    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.25])
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[0, 2])

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
    ax_plot.set_title(f"Spatial Convergence — {case_name}\n({obstacle_type.capitalize()}, Re={Re})", fontsize=13, fontweight='bold')
    ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_plot.legend(fontsize=10, framealpha=0.9)

    # 2. Vertical Velocity Profile
    ax_prof.plot(u_mag_ref, y_ref, 'k-', lw=2.2, label='Conforming (ref.)', zorder=5)

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(resolutions)))
    for (n, color) in zip(resolutions, colors):
        y_p, u_p = profiles[n]
        ax_prof.plot(u_p, y_p, '--', color=color, lw=1.5, label=f'n = {n}')

    ax_prof.set_xlabel(r"$\|\mathbf{u}\|$ [m/s]", fontsize=11)
    ax_prof.set_ylabel("$y$ [m]", fontsize=11)
    ax_prof.set_title(f"Centerline Profile\n($x = {x_obs:.2f}, t = T_{{final}}$)", fontsize=12, fontweight='bold')
    ax_prof.grid(True, ls="--", alpha=0.5)
    ax_prof.legend(fontsize=9, loc='upper right')

    # 3. Convergence Summary Tables
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
