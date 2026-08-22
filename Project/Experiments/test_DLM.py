"""
Numerical Experiment: Upstream Buffer Layer Dirichlet Recovery via Distributed Lagrange Multipliers (DLM)
Phase 1 (Conforming Benchmark on Omega_0) and Phase 2 (Buffer Recovery via DLM).

Solves the flow equations directly using the project's Conforming_solver and NS_DLM_Solver classes
without re-implementing the variational formulations or manual Stokes warm-up routines.
"""

import os
import sys
import gc
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

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, sqrt,
    assemble, conditional, lt, ge, sin, cos, pi
)

from Solvers.NS_Conforming import Conforming_solver
from Solvers.NS_DLM_simple import NS_DLM_Solver


# =============================================================================
# 1. MMS EXACT SOLUTION & FORCING DEFINITION
# =============================================================================

class ManufacturedSolution:
    """Exact divergence-free analytical solution and corresponding Navier-Stokes body force."""
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = 0.2  # Characteristic scale
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
        """Analytical momentum source term: f = rho*(u.grad)u - div(2*mu*sym(grad(u))) + grad(p)."""
        X = SpatialCoordinate(mesh)
        u_ex = self.u_exact(mesh)
        p_ex = self.p_exact(mesh)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press


class BufferObstacle:
    """Obstacle representing the upstream buffer layer region (x < 0)."""
    def __init__(self, L_buf: float = 1.0):
        self.L_buf = L_buf

    def chi(self, mesh, t=None):
        X = SpatialCoordinate(mesh)
        return conditional(lt(X[0], 0.0), 1.0, 0.0)

    def distExpr(self, mesh, t=None):
        X = SpatialCoordinate(mesh)
        return X[0]

    def us_x(self, t=None):
        return Constant(0.0)

    def us_y(self, t=None):
        return Constant(0.0)

    def get_characteristic_length(self):
        return 1.0


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
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 3. PHASE 2: BUFFER RECOVERY SOLVER (Omega_buf + Omega_0 with DLM)
# =============================================================================

def solve_phase2_dlm_buffer(n: int, mms: ManufacturedSolution,
                            Lx: float = 4.0, Ly: float = 1.0, L_buf: float = 1.0,
                            T_end: float = 5.0, dt: float = 0.5):
    """
    Solves Navier-Stokes on extended domain [-L_buf, Lx] x [0, Ly] using NS_DLM_Solver.
    """
    nx_buf = int(round(L_buf * n))
    nx_phys = int(round(Lx * n))
    n_tot = nx_buf + nx_phys
    ny = int(round(Ly * n))
    L_tot = L_buf + Lx

    # Fluid mesh on extended domain
    fluid_mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
    fluid_mesh.coordinates.dat.data[:, 0] -= L_buf

    # Solid mesh on buffer region [-L_buf, 0] x [0, Ly]
    solid_mesh = RectangleMesh(nx_buf, ny, L_buf, Ly)
    solid_mesh.coordinates.dat.data[:, 0] -= L_buf

    buf_obstacle = BufferObstacle(L_buf=L_buf)
    solver = NS_DLM_Solver(moving=False, n=n, Re=mms.Re)

    mesh_out, uh, ph = solver.NS_DLM_Solve(
        fluid_mesh=fluid_mesh,
        solid_mesh=solid_mesh,
        obstacle=buf_obstacle,
        f_custom=mms.f_forcing,
        u_exact=mms.u_exact,
        p_exact=mms.p_exact,
        dt=dt,
        t_final=T_end
    )
    return uh, ph, mesh_out


# =============================================================================
# 4. ERROR EVALUATION & PROFILE EXTRACTION
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
    """Computes velocity L2, H1 and pressure L2 errors restricted strictly to Omega_0 (x >= 0)."""
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
    """Extracts vertical velocity cut u_x at the physical interface Sigma (x = 0)."""
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
# 5. EXPERIMENTAL CONVERGENCE PIPELINE
# =============================================================================

def run_dlm_experiment_pipeline(
    resolutions: List[int] = [75, 100, 125],
    Lx: float = 4.0,
    Ly: float = 1.0,
    L_buf: float = 1.0,
    Re: float = 40.0,
    T_end: float = 5.0,
    dt: float = 0.5,
    output_dir: str = "results_dlm_buffer_recovery"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    print("=" * 90)
    print("UPSTREAM BUFFER RECOVERY EXPERIMENT: DLM BUFFER")
    print(f"Domain: Physical [0, {Lx}] x [0, {Ly}] + Buffer [-{L_buf}, 0] x [0, {Ly}] | Re = {Re}")
    print(f"Mesh Resolutions n: {resolutions} | Final Time T: {T_end}s (dt = {dt}s)")
    print("=" * 90)

    res_p1 = {"L2_u": [], "H1_u": [], "L2_p": []}
    res_p2 = {"L2_u": [], "H1_u": [], "L2_p": [], "interf_L2": []}
    
    profiles_p2: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    h_vals = [1.0 / n for n in resolutions]

    for n in resolutions:
        print(f"\n---> Running Resolution n = {n:3d} (h = {1.0/n:.4f}) ...")

        # 1. Phase 1: Conforming Benchmark
        uh_1, ph_1, mesh_1 = solve_phase1_conforming(n, mms, Lx, Ly, T_end, dt)
        e_L2_u1, e_H1_u1, e_L2_p1 = compute_errors_phase1(mesh_1, uh_1, ph_1, mms)
        res_p1["L2_u"].append(e_L2_u1)
        res_p1["H1_u"].append(e_H1_u1)
        res_p1["L2_p"].append(e_L2_p1)
        print(f"  [Phase 1 Conforming] L2(u): {e_L2_u1:.5e} | H1(u): {e_H1_u1:.5e} | L2(p): {e_L2_p1:.5e}")

        # 2. Phase 2: DLM Buffer Recovery
        uh_2, ph_2, mesh_2 = solve_phase2_dlm_buffer(n, mms, Lx, Ly, L_buf, T_end, dt)
        e_L2_u2, e_H1_u2, e_L2_p2 = compute_errors_phase2_restricted(mesh_2, uh_2, ph_2, mms)
        y_pts, u_x_num, u_x_ex = extract_interface_profile(uh_2, mms)
        e_intf = float(np.sqrt(_trapezoid((u_x_num - u_x_ex)**2, y_pts)))

        res_p2["L2_u"].append(e_L2_u2)
        res_p2["H1_u"].append(e_H1_u2)
        res_p2["L2_p"].append(e_L2_p2)
        res_p2["interf_L2"].append(e_intf)
        profiles_p2[n] = (y_pts, u_x_num)
        print(f"  [Phase 2 DLM Buffer] L2(u): {e_L2_u2:.5e} | H1(u): {e_H1_u2:.5e} | L2(p): {e_L2_p2:.5e} | Interface L2: {e_intf:.5e}")

        del uh_1, ph_1, mesh_1, uh_2, ph_2, mesh_2
        gc.collect()

    # Compute empirical spatial convergence rates
    def compute_rates(err_list, h_list):
        return [np.log(err_list[i] / err_list[i+1]) / np.log(h_list[i] / h_list[i+1]) for i in range(len(h_list)-1)]

    rates_p1_L2 = compute_rates(res_p1["L2_u"], h_vals)
    rates_p1_H1 = compute_rates(res_p1["H1_u"], h_vals)
    rates_p2_L2 = compute_rates(res_p2["L2_u"], h_vals)
    rates_p2_H1 = compute_rates(res_p2["H1_u"], h_vals)

    # Print summary tables
    print("\n" + "=" * 85)
    print("Table 1: Phase 1 - BENCHMARK CONFORMING (Omega_0)")
    print("=" * 85)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'L2(p) Error':>14}")
    print("-" * 85)
    for i, n in enumerate(resolutions):
        r_l2 = f"{rates_p1_L2[i-1]:+5.2f}" if i > 0 else "   -- "
        r_h1 = f"{rates_p1_H1[i-1]:+5.2f}" if i > 0 else "   -- "
        print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p1['L2_u'][i]:14.5e} | {r_l2} | {res_p1['H1_u'][i]:14.5e} | {r_h1} | {res_p1['L2_p'][i]:14.5e}")

    print("\n" + "=" * 95)
    print("Table 2: Phase 2 - UPSTREAM BUFFER RECOVERY VIA DLM (Restricted to Omega_0)")
    print("=" * 95)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'Interface L2':>14}")
    print("-" * 95)
    for i, n in enumerate(resolutions):
        r_l2 = f"{rates_p2_L2[i-1]:+5.2f}" if i > 0 else "   -- "
        r_h1 = f"{rates_p2_H1[i-1]:+5.2f}" if i > 0 else "   -- "
        print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p2['L2_u'][i]:14.5e} | {r_l2} | {res_p2['H1_u'][i]:14.5e} | {r_h1} | {res_p2['interf_L2'][i]:14.5e}")
    print("=" * 95 + "\n")

    # =========================================================================
    # 6. GENERATE COMPARISON PLOTS
    # =========================================================================

    # Plot 1: Spatial Log-Log Convergence Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Spatial Convergence: Pure Conforming vs DLM Buffer Recovery", fontsize=13, fontweight='bold')

    h_arr = np.array(h_vals)
    ax1.loglog(h_arr, res_p1["L2_u"], 'o-', color='#1f77b4', linewidth=2, label='Phase 1: Conforming $L^2(\\mathbf{u})$')
    ax1.loglog(h_arr, res_p2["L2_u"], 's--', color='#d62728', linewidth=2, label='Phase 2: DLM Buffer $L^2(\\mathbf{u})$')
    ax1.loglog(h_arr, [res_p1["L2_u"][0] * (h / h_vals[0])**2 for h in h_arr], 'k:', alpha=0.6, label='Optimal $O(h^2)$')
    ax1.loglog(h_arr, [res_p1["L2_u"][0] * (h / h_vals[0])**1 for h in h_arr], 'k--', alpha=0.6, label='Slope $O(h)$')
    ax1.set_xlabel("Mesh Size $h = 1/n$", fontsize=11)
    ax1.set_ylabel("Velocity $L^2$ Error in $\\Omega_0$", fontsize=11)
    ax1.set_title("Velocity $L^2$ Error Norm", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    ax2.loglog(h_arr, res_p1["H1_u"], 'o-', color='#1f77b4', linewidth=2, label='Phase 1: Conforming $H^1(\\mathbf{u})$')
    ax2.loglog(h_arr, res_p2["H1_u"], 's--', color='#d62728', linewidth=2, label='Phase 2: DLM Buffer $H^1(\\mathbf{u})$')
    ax2.loglog(h_arr, [res_p1["H1_u"][0] * (h / h_vals[0])**1 for h in h_arr], 'k--', alpha=0.6, label='Optimal $O(h)$')
    ax2.set_xlabel("Mesh Size $h = 1/n$", fontsize=11)
    ax2.set_ylabel("Velocity $H^1$ Error in $\\Omega_0$", fontsize=11)
    ax2.set_title("Velocity $H^1$ Error Norm", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)

    plt.tight_layout()
    conv_plot_path = os.path.join(output_dir, "convergence_comparison_loglog.png")
    plt.savefig(conv_plot_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Log-Log Convergence Plot: {conv_plot_path}")

    # Plot 2: Interface Velocity Cut Profile
    fig_prof, ax_prof = plt.subplots(figsize=(7, 6))
    fig_prof.suptitle("Velocity Profile Recovery at Interface $\\Sigma$ ($x = 0$)", fontsize=12, fontweight='bold')

    y_fine = np.linspace(0, Ly, 200)
    u_ex_line = np.ones_like(y_fine)  # target u_x at x=0 is 1.0
    ax_prof.plot(u_ex_line, y_fine, 'k-', linewidth=2.5, label='Target Dirichlet $\\mathbf{u}_{ex}(0,y) = 1.0$')

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(resolutions)))
    for idx, n in enumerate(resolutions):
        y_p, u_p = profiles_p2[n]
        ax_prof.plot(u_p, y_p, '--', color=colors[idx], linewidth=1.8, label=f'DLM Rec. (n={n})')

    ax_prof.set_xlabel("Horizontal Velocity $u_x(0, y)$", fontsize=11)
    ax_prof.set_ylabel("Channel Height $y$", fontsize=11)
    ax_prof.set_xlim(0.8, 1.2)
    ax_prof.set_ylim(0, Ly)
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    prof_plot_path = os.path.join(output_dir, "interface_velocity_recovery.png")
    plt.savefig(prof_plot_path, dpi=300)
    plt.close(fig_prof)
    print(f">> Saved DLM Interface Profile Plot: {prof_plot_path}")


# =============================================================================
# 7. EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_dlm_experiment_pipeline(
        resolutions=[20, 40, 60],
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,
        Re=40.0,
        T_end=2.0,
        dt=0.2,
        output_dir="results_dlm_buffer_recovery"
    )