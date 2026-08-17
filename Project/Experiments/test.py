"""
Numerical Experiment: Upstream Buffer Layer Dirichlet Recovery via Brinkman Penalization
"""

import os
import sys
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, VectorFunctionSpace, FunctionSpace, Function,
    TrialFunctions, TestFunctions, DirichletBC, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, ds, sqrt,
    assemble, solve, conditional, lt, ge, project, sin, cos, pi
)


# =============================================================================
# 1. MMS EXACT SOLUTION & FORCING DEFINITION
# =============================================================================

class ManufacturedSolution:
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = 0.2  # Characteristic length for Reynolds number
        self.mu = self.rho * self.L_char * self.u_char / self.Re

    def u_exact(self, X):
        x, y = X[0], X[1]
        u_x = sin(pi * x / self.Lx) * sin(2.0 * pi * y / self.Ly)
        u_y = (self.Ly / (2.0 * self.Lx)) * cos(pi * x / self.Lx) * (cos(2.0 * pi * y / self.Ly) - 1.0)
        return as_vector([u_x, u_y])

    def p_exact(self, X):
        x, y = X[0], X[1]
        return sin(pi * x / self.Lx) * sin(pi * y / self.Ly)

    def f_forcing(self, X):
        """Compute the force of the exact solution for the Navier-Stokes problem"""
        u_ex = self.u_exact(X)
        p_ex = self.p_exact(X)
        
        # f = rho * (u . grad) u - div(2 * mu * sym(grad(u))) + grad(p)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press


# =============================================================================
# 2. FASE 1: CONFORMING BENCHMARK SOLVER (Dominio Fisico Omega_0)
# =============================================================================

def solve_phase1_conforming(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                            T_end: float = 5.0, dt: float = 0.5) -> Tuple[Function, Function, object]:
    """
    Solve the NS problem on the first domain Omega_0 = [0, Lx] x [0, Ly] with exact Dirichlet boundary conditions.
    """
    ny = max(4, int(round(n * Ly / Lx)))
    mesh = RectangleMesh(n, ny, Lx, Ly)
    X = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    u_ex = mms.u_exact(X)
    p_ex = mms.p_exact(X)
    f_val = mms.f_forcing(X)

    bcs = [
        DirichletBC(W.sub(0), u_ex, 1),
        DirichletBC(W.sub(0), u_ex, 2),
        DirichletBC(W.sub(0), u_ex, 3),
        DirichletBC(W.sub(0), u_ex, 4),
        # DirichletBC(W.sub(1), p_ex, 4) 
    ]

    uh_n = Function(V)
    # uh_n.assign(0.0)
    uh_n.interpolate(u_ex)
    sol = Function(W)
    uh, ph = sol.subfunctions

    # Variational formulation of the problem
    a = (Constant(mms.rho) / Constant(dt)) * inner(u, v) * dx \
        + Constant(mms.rho) * inner(dot(uh_n, nabla_grad(u)), v) * dx \
        + 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
        - div(v) * p * dx \
        + div(u) * q * dx

    L = (Constant(mms.rho) / Constant(dt)) * inner(uh_n, v) * dx + inner(f_val, v) * dx

    num_steps = max(1, int(round(T_end / dt)))
    for _ in range(num_steps):
        solve(a == L, sol, bcs=bcs,
              solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
        uh_n.assign(uh)

    return uh, ph, mesh


# =============================================================================
# 3. FASE 2: BUFFER RECOVERY SOLVER (Omega_buf + Omega_0 with Brinkman)
# =============================================================================

def solve_phase2_brinkman_buffer(n: int, mms: ManufacturedSolution, Lx: float = 4.0, Ly: float = 1.0,
                                 L_buf: float = 1.0, R_penalty: float = 1.0e5,
                                 T_end: float = 5.0, dt: float = 0.5) -> Tuple[Function, Function, object]:
    """
    Solves the extended problem on [-L_buf, Lx] x [0, Ly].
    - x in [-L_buf, 0): buffer region Omega_buf with Brinkman to u_exact.
    - x in [0, Lx]: physical region Omega_0.
    - Boundary x = -L_buf (Tag 3): Neumann condition.
    """

    nx_buf = int(round(L_buf * n))
    nx_phys = int(round(Lx * n))
    n_tot = nx_buf + nx_phys
    ny = int(round(Ly * n))
    L_tot = L_buf + Lx

    mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
    mesh.coordinates.dat.data[:, 0] -= L_buf

    X = SpatialCoordinate(mesh)
    x = X[0]

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    u_ex = mms.u_exact(X)
    p_ex = mms.p_exact(X)
    f_val = mms.f_forcing(X)

    chi_buf = conditional(lt(x, 0.0), 1.0, 0.0)

    # BCs for the extended domain:
    bcs = [
        DirichletBC(W.sub(0), u_ex, 1),
        DirichletBC(W.sub(0), u_ex, 2),
        DirichletBC(W.sub(0), u_ex, 4),
        # DirichletBC(W.sub(1), p_ex, 4)
    ]

    uh_n = Function(V)
    # uh_n.assign(0.0)
    uh_n.interpolate(u_ex)
    sol = Function(W)
    uh, ph = sol.subfunctions

    # Variational Formulation (Brinkman)
    a = (Constant(mms.rho) / Constant(dt)) * inner(u, v) * dx \
        + Constant(mms.rho) * inner(dot(uh_n, nabla_grad(u)), v) * dx \
        + 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
        - div(v) * p * dx \
        + div(u) * q * dx \
        + Constant(R_penalty) * chi_buf * inner(u, v) * dx

    L = (Constant(mms.rho) / Constant(dt)) * inner(uh_n, v) * dx \
        + inner(f_val, v) * dx \
        + Constant(R_penalty) * chi_buf * inner(u_ex, v) * dx

    num_steps = max(1, int(round(T_end / dt)))
    for _ in range(num_steps):
        solve(a == L, sol, bcs=bcs,
              solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
        uh_n.assign(uh)

    return uh, ph, mesh


# =============================================================================
# 4. ERROR EVALUATION & INTERFACE EXTRACTION
# =============================================================================

def compute_errors_phase1(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Compute the errors L2(u), H1(u) and L2(p) on Omega_0."""
    X = SpatialCoordinate(mesh)
    u_ex = mms.u_exact(X)
    p_ex = mms.p_exact(X)

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
    u_ex = mms.u_exact(X)
    p_ex = mms.p_exact(X)

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
    """Exctract the vertical profile of velocity u_x along the interface Sigma (x = 0)."""
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
            u_exact_x[i] = math.sin(0.0) * math.sin(2.0 * math.pi * y_val / mms.Ly)

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
    output_dir: str = "buffer_experiment_results"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    print("=" * 80)
    print("UPSTREAM BUFFER RECOVERY EXPERIMENT: CONFORMING vs BRINKMAN BUFFER")
    print(f"Domain: Physical [0, {Lx}] x [0, {Ly}] | Buffer length: {L_buf} | Re: {Re} | R: {R_penalty:.1e}")
    print(f"Resolutions n: {resolutions} | Final Time T: {T_end}s (dt = {dt}s)")
    print("=" * 80)

    res_p1 = {"L2_u": [], "H1_u": [], "L2_p": []}
    res_p2 = {"L2_u": [], "H1_u": [], "L2_p": [], "interf_L2": []}
    
    profiles_p2 = {}
    h_vals = [1.0 / n for n in resolutions]

    # --- Loop of simulation on every resolution ---
    for n in resolutions:
        print(f"\n---> Running Resolution n = {n} (h = {1.0/n:.4f})")
        
        # 1. Phase 1: Conforming
        uh_1, ph_1, mesh_1 = solve_phase1_conforming(n, mms, Lx, Ly, T_end, dt)
        e_L2_u1, e_H1_u1, e_L2_p1 = compute_errors_phase1(mesh_1, uh_1, ph_1, mms)
        res_p1["L2_u"].append(e_L2_u1)
        res_p1["H1_u"].append(e_H1_u1)
        res_p1["L2_p"].append(e_L2_p1)
        print(f"  [Phase 1 Conforming] L2(u): {e_L2_u1:.4e} | H1(u): {e_H1_u1:.4e} | L2(p): {e_L2_p1:.4e}")

        # 2. Phase 2: Buffer Brinkman
        uh_2, ph_2, mesh_2 = solve_phase2_brinkman_buffer(n, mms, Lx, Ly, L_buf, R_penalty, T_end, dt)
        e_L2_u2, e_H1_u2, e_L2_p2 = compute_errors_phase2_restricted(mesh_2, uh_2, ph_2, mms)
        
        # Interface profile
        y_pts, u_num_x, u_ex_x = extract_interface_profile(uh_2, mms)
        profiles_p2[n] = (y_pts, u_num_x)
        trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
        e_interf_L2 = np.sqrt(trapezoid((u_num_x - u_ex_x)**2, y_pts))
        
        res_p2["L2_u"].append(e_L2_u2)
        res_p2["H1_u"].append(e_H1_u2)
        res_p2["L2_p"].append(e_L2_p2)
        res_p2["interf_L2"].append(e_interf_L2)
        print(f"  [Phase 2 Buffer Rec] L2(u): {e_L2_u2:.4e} | H1(u): {e_H1_u2:.4e} | L2(p): {e_L2_p2:.4e} | Intf_L2(x=0): {e_interf_L2:.4e}")

    # --- Convergence rates ---
    def compute_rates(err_list, h_list):
        return [np.log(err_list[i] / err_list[i+1]) / np.log(h_list[i] / h_list[i+1]) for i in range(len(h_list)-1)]

    rates_p1_L2 = compute_rates(res_p1["L2_u"], h_vals)
    rates_p1_H1 = compute_rates(res_p1["H1_u"], h_vals)
    rates_p2_L2 = compute_rates(res_p2["L2_u"], h_vals)
    rates_p2_H1 = compute_rates(res_p2["H1_u"], h_vals)

    # --- tables ---
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
    print("Table 2: Phase 2 - UPSTREAM BUFFER RECOVERY VIA BRINKMAN (Restricted to Omega_0)")
    print("=" * 95)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'Rate':>6} | {'H1(u) Error':>14} | {'Rate':>6} | {'Interface L2':>14}")
    print("-" * 95)
    for i, n in enumerate(resolutions):
        r_l2 = f"{rates_p2_L2[i-1]:+5.2f}" if i > 0 else "   -- "
        r_h1 = f"{rates_p2_H1[i-1]:+5.2f}" if i > 0 else "   -- "
        print(f"{n:5d} | {h_vals[i]:8.4f} | {res_p2['L2_u'][i]:14.5e} | {r_l2} | {res_p2['H1_u'][i]:14.5e} | {r_h1} | {res_p2['interf_L2'][i]:14.5e}")
    print("=" * 95 + "\n")

    # =========================================================================
    # 6. Generate the plots to compare the solutions
    # =========================================================================
    
    # Figure 1: Convergence plot Log-Log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Spatial Convergence: Pure Conforming vs Brinkman Buffer Recovery", fontsize=13, fontweight='bold')

    # L2 Velocity Plot
    ax1.loglog(h_vals, res_p1["L2_u"], 'o-', color='#1f77b4', linewidth=2, label='Phase 1: Conforming $L^2(\\mathbf{u})$')
    ax1.loglog(h_vals, res_p2["L2_u"], 's--', color='#d62728', linewidth=2, label='Phase 2: Brinkman Buffer $L^2(\\mathbf{u})$')
    ax1.loglog(h_vals, [res_p1["L2_u"][0] * (h / h_vals[0])**2 for h in h_vals], 'k:', alpha=0.6, label='Optimal $O(h^2)$')
    ax1.loglog(h_vals, [res_p1["L2_u"][0] * (h / h_vals[0])**1 for h in h_vals], 'k--', alpha=0.6, label='Slope $O(h)$')
    ax1.set_xlabel("Mesh Size $h = 1/n$", fontsize=11)
    ax1.set_ylabel("Velocity $L^2$ Error in $\\Omega_0$", fontsize=11)
    ax1.set_title("Velocity $L^2$ Error Norm", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    # H1 Velocity Plot
    ax2.loglog(h_vals, res_p1["H1_u"], 'o-', color='#1f77b4', linewidth=2, label='Phase 1: Conforming $H^1(\\mathbf{u})$')
    ax2.loglog(h_vals, res_p2["H1_u"], 's--', color='#d62728', linewidth=2, label='Phase 2: Brinkman Buffer $H^1(\\mathbf{u})$')
    ax2.loglog(h_vals, [res_p1["H1_u"][0] * (h / h_vals[0])**1 for h in h_vals], 'k--', alpha=0.6, label='Optimal $O(h)$')
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

    # FIGURE 2: Recover of the velocity profile x = 0
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Velocity Profile Recovery at Interface $\\Sigma$ ($x = 0$)", fontsize=12, fontweight='bold')
    
    # Anallytical profile at x = 0
    y_fine = np.linspace(0, Ly, 200)
    u_ex_line = np.zeros_like(y_fine)  # u_x esatta a x=0 è identicamente 0
    ax.plot(u_ex_line, y_fine, 'k-', linewidth=2.5, label='Target Dirichlet $\\mathbf{u}_{ex}(0,y) = 0$')

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(resolutions)))
    for idx, n in enumerate(resolutions):
        y_p, u_p = profiles_p2[n]
        ax.plot(u_p, y_p, '--', color=colors[idx], linewidth=1.8, label=f'Brinkman Rec. (n={n})')

    ax.set_xlabel("Horizontal Velocity $u_x(0, y)$", fontsize=11)
    ax.set_ylabel("Channel Height $y$", fontsize=11)
    ax.set_xlim(-0.05, 0.25)
    ax.set_ylim(0, Ly)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    profile_plot_path = os.path.join(output_dir, "interface_velocity_recovery.png")
    plt.savefig(profile_plot_path, dpi=300)
    plt.close(fig)
    print(f">> Saved Interface Profile Recovery Plot: {profile_plot_path}")


# =============================================================================
# 7. EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_experiment_pipeline(
        resolutions=[75, 100, 125],    # Risoluzioni n (elementi per lato lungo di Omega_0)
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,                      # Lunghezza del buffer a monte Omega_buf
        Re=40.0,
        R_penalty=1.0e4,                # Resistenza idraulica Brinkman
        T_end=5.0,                      # Tempo finale di simulazione
        dt=0.5,
        output_dir="results_buffer_recovery"
    )