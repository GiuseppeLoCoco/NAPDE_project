"""
Plotting utilities for validation and convergence benchmarks.
Centralizes all visualization functions (convergence curves, field comparisons, profiles, and summary tables).
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from firedrake import FunctionSpace, Function
from firedrake.pyplot import tripcolor, tricontour, tricontourf, triplot
import matplotlib.patches as patches


# =============================================================================
# 1. L2 PENALIZATION PAPER PLOTS (Angot et al. 1999, Section 6.1)
# =============================================================================

def plot_l2_penalization_convergence(results, output_dir: str):
    """
    1. Plots log-log convergence of L2 velocity error norms in solid and fluid vs penalization parameter eta (Table 1).
    """
    etas = [r['eta'] for r in results]
    err_s = [r['err_solid'] for r in results]
    err_f = [r['err_fluid'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.loglog(etas, err_s, 'o-b', linewidth=2, label=r'$\|u_\eta\|_{L^2(\Omega_s)}$ (Solid error)')
    ax.loglog(etas, err_f, 's-r', linewidth=2, label=r'$\|u_\eta - u_{ref}\|_{L^2(\Omega_f)}$ (Fluid error)')
    ax.loglog(etas, [etas[0]**1.0 * err_s[0] / (etas[0]**1.0) * (e/etas[0]) for e in etas], '--k', label=r'Theoretical $O(\eta)$')
    
    ax.set_xlabel(r'Penalization parameter $\eta$', fontsize=12)
    ax.set_ylabel(r'$L^2$ Error Norm', fontsize=12)
    ax.set_title('Convergence of L2 Penalization Method at Re = 40 (Table 1)', fontsize=13, fontweight='bold')
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=11)
    
    plot_path_conv = os.path.join(output_dir, "l2_error_convergence_re40.png")
    fig.savefig(plot_path_conv, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n1. Saved Convergence plot to: {plot_path_conv}")


def plot_pressure_comparison(ref_mesh, p_ref, pen_mesh, p_pen, eta: float, output_dir: str,
                             x_obs: float = 1.0, y_obs: float = 0.5, side_length: float = 0.2):
    """
    2. Plots side-by-side comparison of Pressure: Continuous Colormap + Isobars (contour lines)
    for Reference Conforming vs L2 Penalization.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True, facecolor='white')

    p_ref_arr = p_ref.dat.data_ro
    p_pen_arr = p_pen.dat.data_ro
    p_min = min(float(np.min(p_ref_arr)), float(np.min(p_pen_arr)))
    p_max = max(float(np.max(p_ref_arr)), float(np.max(p_pen_arr)))
    levels = np.linspace(p_min, p_max, 28)

    obs_x0 = x_obs - side_length / 2.0
    obs_y0 = y_obs - side_length / 2.0

    # 1. Reference Pressure + Isobars
    axes[0].set_title("Reference (Conforming Dirichlet): Pressure Field & Isobars", fontsize=11, fontweight='bold')
    p0 = tripcolor(p_ref, axes=axes[0], cmap='RdYlBu_r', shading='gouraud', vmin=p_min, vmax=p_max)
    tricontour(p_ref, axes=axes[0], levels=levels, colors='black', linewidths=0.55, alpha=0.7)
    fig.colorbar(p0, ax=axes[0], fraction=0.02, pad=0.02, label='p')
    rect_ref = patches.Rectangle((obs_x0, obs_y0), side_length, side_length,
                                 linewidth=1.5, edgecolor='black', facecolor='lightgray', zorder=10)
    axes[0].add_patch(rect_ref)

    # 2. L2 Penalization Pressure + Isobars
    axes[1].set_title(fr"L2 Penalization ($\eta = {eta:.1e}$, $R = {1.0/eta:.1e}$): Pressure Field & Isobars", fontsize=11, fontweight='bold')
    p1 = tripcolor(p_pen, axes=axes[1], cmap='RdYlBu_r', shading='gouraud', vmin=p_min, vmax=p_max)
    tricontour(p_pen, axes=axes[1], levels=levels, colors='black', linewidths=0.55, alpha=0.7)
    fig.colorbar(p1, ax=axes[1], fraction=0.02, pad=0.02, label=r'$p_\eta$')
    rect_pen = patches.Rectangle((obs_x0, obs_y0), side_length, side_length,
                                 linewidth=1.8, edgecolor='red', facecolor='none', linestyle='--', zorder=10,
                                 label=r'Immersed $\partial\Omega_s$')
    axes[1].add_patch(rect_pen)
    axes[1].legend(loc='upper right', fontsize=10)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("$y$", fontsize=11)

    axes[1].set_xlabel("$x$", fontsize=11)

    plt.tight_layout()
    plot_path_press = os.path.join(output_dir, "fig3_pressure_comparison.png")
    fig.savefig(plot_path_press, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"2. Saved Pressure Comparison plot to: {plot_path_press}")


def plot_vorticity_comparison(ref_mesh, u_ref, pen_mesh, u_pen, eta: float, output_dir: str,
                              x_obs: float = 1.0, y_obs: float = 0.5, side_length: float = 0.2):
    """
    3. Plots side-by-side comparison of Vorticity: Continuous Colormap + Isolines (contour lines)
    for Reference Conforming vs L2 Penalization with symmetric colormap.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True, facecolor='white')

    # Compute vorticity: curl(u) = du_y/dx - du_x/dy
    V_scalar_ref = FunctionSpace(ref_mesh, "CG", 1)
    vort_ref = Function(V_scalar_ref, name="Vorticity_ref").interpolate(u_ref[1].dx(0) - u_ref[0].dx(1))

    V_scalar_pen = FunctionSpace(pen_mesh, "CG", 1)
    vort_pen = Function(V_scalar_pen, name="Vorticity_pen").interpolate(u_pen[1].dx(0) - u_pen[0].dx(1))

    v_ref_arr = vort_ref.dat.data_ro
    v_pen_arr = vort_pen.dat.data_ro
    v_lim = max(float(np.percentile(np.abs(v_ref_arr), 98)), float(np.percentile(np.abs(v_pen_arr), 98)))
    v_lim = max(1.0, v_lim)
    levels = np.linspace(-v_lim, v_lim, 28)

    obs_x0 = x_obs - side_length / 2.0
    obs_y0 = y_obs - side_length / 2.0

    # 1. Reference Vorticity + Isolines
    axes[0].set_title(r"Reference (Conforming Dirichlet): Vorticity Field & Isolines $\omega$", fontsize=11, fontweight='bold')
    p0 = tripcolor(vort_ref, axes=axes[0], cmap='coolwarm', shading='gouraud', vmin=-v_lim, vmax=v_lim)
    tricontour(vort_ref, axes=axes[0], levels=levels, colors='black', linewidths=0.55, alpha=0.6)
    fig.colorbar(p0, ax=axes[0], fraction=0.02, pad=0.02, label=r'$\omega$')
    rect_ref = patches.Rectangle((obs_x0, obs_y0), side_length, side_length,
                                 linewidth=1.5, edgecolor='black', facecolor='lightgray', zorder=10)
    axes[0].add_patch(rect_ref)

    # 2. L2 Penalization Vorticity + Isolines
    axes[1].set_title(fr"L2 Penalization ($\eta = {eta:.1e}$, $R = {1.0/eta:.1e}$): Vorticity Field & Isolines $\omega_\eta$", fontsize=11, fontweight='bold')
    p1 = tripcolor(vort_pen, axes=axes[1], cmap='coolwarm', shading='gouraud', vmin=-v_lim, vmax=v_lim)
    tricontour(vort_pen, axes=axes[1], levels=levels, colors='black', linewidths=0.55, alpha=0.6)
    fig.colorbar(p1, ax=axes[1], fraction=0.02, pad=0.02, label=r'$\omega_\eta$')
    rect_pen = patches.Rectangle((obs_x0, obs_y0), side_length, side_length,
                                 linewidth=1.8, edgecolor='red', facecolor='none', linestyle='--', zorder=10,
                                 label=r'Immersed $\partial\Omega_s$')
    axes[1].add_patch(rect_pen)
    axes[1].legend(loc='upper right', fontsize=10)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("$y$", fontsize=11)

    axes[1].set_xlabel("$x$", fontsize=11)

    plt.tight_layout()
    plot_path_vort = os.path.join(output_dir, "fig3_vorticity_comparison.png")
    fig.savefig(plot_path_vort, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"3. Saved Vorticity Comparison plot to: {plot_path_vort}")


def plot_strouhal_comparison(strouhal_data: dict, output_dir: str):
    """
    Plots Unsteady Flow Validation (Angot et al. 1999 Table 2 & Fig. 4):
    Left: Downstream wake velocity signals u_y(t) showing vortex shedding and convective delay.
    Right: Strouhal number comparison St = f*D/U across penalization parameters eta vs Conforming Reference.
    """
    fig, (ax_sig, ax_bar) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor='white')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels = []
    st_vals = []
    bar_colors = []

    for i, (key, data) in enumerate(strouhal_data.items()):
        t = data.get('t', [])
        sig = data.get('uy', [])
        st = data.get('St', 0.0)
        lbl = data.get('label', str(key))
        col = colors[i % len(colors)]

        # Signal plot
        if len(t) > 1 and len(sig) > 1:
            ax_sig.plot(t, sig, label=lbl, color=col, linewidth=1.8, alpha=0.85)

        labels.append(lbl)
        st_vals.append(st)
        bar_colors.append(col)

    # Signal panel styling
    ax_sig.set_xlabel("Time $t$ [s]", fontsize=11)
    ax_sig.set_ylabel(r"Transverse velocity $u_y(t)$ at wake probe", fontsize=11)
    ax_sig.set_title("Vortex Shedding Time History (Fig. 4)", fontsize=12, fontweight='bold')
    ax_sig.grid(True, linestyle=":", alpha=0.6)
    ax_sig.legend(fontsize=9, loc='upper right')

    # Strouhal Bar Chart styling
    bars = ax_bar.bar(labels, st_vals, color=bar_colors, width=0.45, edgecolor='black', alpha=0.85)
    ax_bar.set_ylabel("Strouhal Number $St = f D / U$", fontsize=11)
    ax_bar.set_title("Strouhal Number $St$ Comparison (Table 2)", fontsize=12, fontweight='bold')
    ax_bar.axhline(0.166, color='black', linestyle='--', linewidth=1.2, label='Angot benchmark ($St \\approx 0.166$)')
    ax_bar.grid(True, axis='y', linestyle=":", alpha=0.6)
    ax_bar.legend(fontsize=9, loc='upper right')

    # Add numeric labels on bars
    for bar, val in zip(bars, st_vals):
        if val > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2.0, val + 0.004,
                        f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path_st = os.path.join(output_dir, "unsteady_strouhal_and_signals.png")
    fig.savefig(plot_path_st, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved Strouhal and Signal comparison plot to: {plot_path_st}")


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
