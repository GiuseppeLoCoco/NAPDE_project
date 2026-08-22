"""
Numerical Experiment - Strategy A:
Brinkman Penalty Parameter (R) Convergence Sweep at Fixed Mesh Resolution.

This script analyzes the asymptotic modeling error of the Brinkman upstream buffer
by varying R over several orders of magnitude on a fixed fine mesh (n = 100).
"""

import gc
import os
import sys
import math
import warnings
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Compatibility for NumPy 1.x and 2.x trapezoidal integration
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, VectorFunctionSpace, FunctionSpace, Function,
    TrialFunctions, TestFunctions, DirichletBC, Constant, SpatialCoordinate,
    as_vector, inner, dot, grad, sym, div, nabla_grad, dx, ds, sqrt,
    assemble, solve, conditional, lt, ge, sin, cos, pi
)

from firedrake import MixedVectorSpaceBasis, VectorSpaceBasis

# =============================================================================
# 1. MMS EXACT SOLUTION & BODY FORCING
# =============================================================================

class ManufacturedSolution:
    """Exact divergence-free analytical solution and corresponding Navier-Stokes body force."""
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = 0.2
        self.mu = self.rho * self.L_char * self.u_char / self.Re

    def u_exact(self, X):
        x, y = X[0], X[1]
        u_x = 1.0 + sin(pi * x / self.Lx) * sin(2.0 * pi * y / self.Ly)
        u_y = (self.Ly / (2.0 * self.Lx)) * cos(pi * x / self.Lx) * (cos(2.0 * pi * y / self.Ly) - 1.0)
        return as_vector([u_x, u_y])

    def p_exact(self, X):
        x, y = X[0], X[1]
        return sin(pi * x / self.Lx) * sin(pi * y / self.Ly)

    def f_forcing(self, X):
        u_ex = self.u_exact(X)
        p_ex = self.p_exact(X)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press


# =============================================================================
# 2. BRINKMAN BUFFER SOLVER
# =============================================================================

def solve_brinkman_buffer(n: int, R_val: float, mms: ManufacturedSolution,
                          Lx: float = 4.0, Ly: float = 1.0, L_buf: float = 1.0,
                          T_end: float = 2.0, dt: float = 0.5) -> Tuple[Function, Function, object]:
    """
    Solves flow on the extended domain [-L_buf, Lx] x [0, Ly] with Brinkman penalization.
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

    # Dirichlet on top/bottom walls (3, 4) and outlet (2). Upstream inlet (1) is left natural (Neumann)
    bcs = [
        DirichletBC(W.sub(0), u_ex, 2),
        DirichletBC(W.sub(0), u_ex, 3),
        DirichletBC(W.sub(0), u_ex, 4)
    ]

    # First strategy of warm-up: Stokes initialization
    
    # -------------------------------------------------------------------------
    # WARM START: Inizialization with Stokes (Linearity)
    # -------------------------------------------------------------------------
    uh_n = Function(V)
    sol_init = Function(W)
    
    a_init = 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
            - div(v) * p * dx \
            + div(u) * q * dx \
            + Constant(R_val) * chi_buf * inner(u, v) * dx
    
    L_init = inner(f_val, v) * dx \
            + Constant(R_val) * chi_buf * inner(u_ex, v) * dx
    
    solve(a_init == L_init, sol_init, bcs=bcs,
        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        form_compiler_parameters={'quadrature_degree': 8})
    
    uh_n.assign(sol_init.subfunctions[0])
    # -------------------------------------------------------------------------
    
    uh_n_ex = Function(V)
    uh_n_ex.interpolate(u_ex)

    # -------------------------------------------------------------------------
    # Error diagnostics: distance of warm-up initial condition from the
    # exact MMS solution and compare with the pure interpolation error
    # -------------------------------------------------------------------------
    # 1. Distance of the warm-up solution (Stokes/Lifting) from the exact MMS solution
    diff_warmup = uh_n - u_ex
    L2_err_warmup = sqrt(assemble(inner(diff_warmup, diff_warmup) * dx(domain=mesh)))
    H1_warmup_sq = inner(grad(diff_warmup), grad(diff_warmup))
    H1_err_warmup = sqrt(assemble(H1_warmup_sq * dx(domain=mesh)))

    # 2. Distance of the pure interpolation from the exact MMS solution
    uh_n_ex = Function(V).interpolate(u_ex)
    diff_interp = uh_n_ex - u_ex
    L2_err_interp = sqrt(assemble(inner(diff_interp, diff_interp) * dx(domain=mesh)))
    H1_interp_sq = inner(grad(diff_interp), grad(diff_interp))
    H1_err_interp = sqrt(assemble(H1_interp_sq * dx(domain=mesh)))

    print(f"      [Diagnostics] L2 Error Warm-up: {float(L2_err_warmup):.4e}")
    print(f"      [Diagnostics] H1 Error Warm-up: {float(H1_err_warmup):.4e}")
    print(f"      [Diagnostics] L2 Error Interpolation: {float(L2_err_interp):.4e}")
    print(f"      [Diagnostics] H1 Error Interpolation: {float(H1_err_interp):.4e}")
    # -------------------------------------------------------------------------

    sol = Function(W)
    uh, ph = sol.subfunctions

    a = (Constant(mms.rho) / Constant(dt)) * inner(u, v) * dx \
        + Constant(mms.rho) * inner(dot(uh_n, nabla_grad(u)), v) * dx \
        + 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
        - div(v) * p * dx \
        + div(u) * q * dx \
        + Constant(R_val) * chi_buf * inner(u, v) * dx

    L = (Constant(mms.rho) / Constant(dt)) * inner(uh_n, v) * dx \
        + inner(f_val, v) * dx \
        + Constant(R_val) * chi_buf * inner(u_ex, v) * dx

    solver_params = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    num_steps = max(1, int(round(T_end / dt)))

    num_steps_max = 300   
    tol_steady = 1e-7  
    t_val = 0.0

    for step in range(num_steps_max):
        t_val += dt
        solve(a == L, sol, bcs=bcs,
                solver_parameters=solver_params,
                form_compiler_parameters={'quadrature_degree': 8})
        
        diff_u = uh - uh_n
        # increment_L2 = sqrt(assemble(inner(diff_u, diff_u) * dx(domain=mesh)))
        increment_H1 = sqrt(assemble((inner(diff_u, diff_u) + inner(grad(diff_u), grad(diff_u))) * dx(domain=mesh)))

        uh_n.assign(uh)
        
        if float(increment_H1) < tol_steady:
            print(f"      [!] Steady-state reach at step {step + 1} (t = {t_val:.2f}) | Increment: {float(increment_H1):.2e}")
            break
    else:
        print(f"      [!!] WARNING: steady state NOT reached after {num_steps_max} steps "
            f"(last increment: {float(increment_H1):.2e})")

    del a, L, bcs, u, p, v, q, uh_n
    gc.collect()

    return uh, ph, mesh


# =============================================================================
# 3. ERROR POST-PROCESSING & PROFILE EXTRACTION
# =============================================================================

def compute_restricted_errors(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Computes L2(u), H1(u), and L2(p) error norms restricted strictly to Omega_0 (x >= 0)."""
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
    """Extracts vertical cut of u_x at the physical interface Sigma (x = 0)."""
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
# 4. SWEEP EXECUTION & PLOTTING PIPELINE
# =============================================================================

def run_r_sweep_analysis(
    R_values: List[float] = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7],
    fixed_n: int = 100,
    Lx: float = 4.0,
    Ly: float = 1.0,
    L_buf: float = 1.0,
    Re: float = 40.0,
    T_end: float = 2.0,
    dt: float = 0.5,
    output_dir: str = "results_strategy_A_R_sweep"
):
    os.makedirs(output_dir, exist_ok=True)
    mms = ManufacturedSolution(Lx=Lx, Ly=Ly, Re=Re)

    print("=" * 90)
    print("STRATEGY A: PENALTY PARAMETER SWEEP (R-SWEEP AT FIXED MESH RESOLUTION)")
    print(f"Fixed Mesh Resolution: n = {fixed_n} (h = {1.0/fixed_n:.4f}) | Re = {Re}")
    print(f"Brinkman Resistance Range R: {R_values}")
    print(f"Simulation Time: T = {T_end}s (dt = {dt}s)")
    print("=" * 90)

    errs_L2_u, errs_H1_u, errs_L2_p, errs_intf = [], [], [], []
    profiles: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for R_val in R_values:
        print(f"\n---> Solving for Brinkman Penalty R = {R_val:.1e} ...")
        uh, ph, mesh = solve_brinkman_buffer(fixed_n, R_val, mms, Lx, Ly, L_buf, T_end, dt)
        
        e_L2, e_H1, e_p = compute_restricted_errors(mesh, uh, ph, mms)
        y_pts, u_x_num, u_x_ex = extract_interface_profile(uh, mms)
        e_intf = float(np.sqrt(_trapezoid((u_x_num - u_x_ex)**2, y_pts)))

        errs_L2_u.append(e_L2)
        errs_H1_u.append(e_H1)
        errs_L2_p.append(e_p)
        errs_intf.append(e_intf)
        profiles[R_val] = (y_pts, u_x_num)

        print(f"     L2(u) in Omega_0: {e_L2:.5e} | H1(u): {e_H1:.5e} | Interface L2: {e_intf:.5e}")

    # Compute convergence rates with respect to 1/R
    rates_L2 = [np.log(errs_L2_u[i] / errs_L2_u[i+1]) / np.log(R_values[i+1] / R_values[i]) for i in range(len(R_values)-1)]
    rates_intf = [np.log(errs_intf[i] / errs_intf[i+1]) / np.log(R_values[i+1] / R_values[i]) for i in range(len(R_values)-1)]

    # Print Summary Table
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

    # Generate Error vs R Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Strategy A: Asymptotic Modeling Error vs Penalty $R$ ($n = {fixed_n}$)", fontsize=13, fontweight='bold')

    R_arr = np.array(R_values)
    ax1.loglog(R_arr, errs_L2_u, 'o-', color='#1f77b4', linewidth=2, label='Velocity $L^2(\\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_H1_u, 's--', color='#ff7f0e', linewidth=1.8, label='Velocity $H^1(\\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_L2_p, '^:', color='#2ca02c', linewidth=1.8, label='Pressure $L^2(\\Omega_0)$ Error')
    ax1.loglog(R_arr, errs_L2_u[0] * (R_arr[0] / R_arr)**0.5, 'k--', alpha=0.5, label='Asymptotic $O(R^{-1/2})$')
    ax1.loglog(R_arr, errs_L2_u[0] * (R_arr[0] / R_arr)**1.0, 'k:', alpha=0.5, label='Asymptotic $O(R^{-1})$')
    ax1.set_xlabel("Brinkman Penalty Parameter $R$", fontsize=11)
    ax1.set_ylabel("Error Norms in Physical Domain $\\Omega_0$", fontsize=11)
    ax1.set_title("Modeling Error in $\\Omega_0$ vs $R$", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.loglog(R_arr, errs_intf, 'd-', color='#d62728', linewidth=2, label='Interface Trace $L^2(\\Sigma)$ Error')
    ax2.loglog(R_arr, errs_intf[0] * (R_arr[0] / R_arr)**0.5, 'k--', alpha=0.5, label='Asymptotic $O(R^{-1/2})$')
    ax2.loglog(R_arr, errs_intf[0] * (R_arr[0] / R_arr)**1.0, 'k:', alpha=0.5, label='Asymptotic $O(R^{-1})$')
    ax2.set_xlabel("Brinkman Penalty Parameter $R$", fontsize=11)
    ax2.set_ylabel("Trace Error at Interface $\\Sigma$ ($x = 0$)", fontsize=11)
    ax2.set_title("Dirichlet Recovery Error vs $R$", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_conv = os.path.join(output_dir, "strategy_A_error_vs_R.png")
    plt.savefig(plot_conv, dpi=300)
    plt.close(fig)
    print(f">> Saved Strategy A Error Plot: {plot_conv}")

    # Generate Interface Profile Recovery Plot
    fig_prof, ax_prof = plt.subplots(figsize=(7, 6))
    fig_prof.suptitle(f"Interface Velocity Profile $u_x(0, y)$ vs $R$ ($n = {fixed_n}$)", fontsize=12, fontweight='bold')
    
    y_fine = np.linspace(0, Ly, 200)
    ax_prof.plot(np.ones_like(y_fine), y_fine, 'k-', linewidth=2.5, label='Target $\\mathbf{u}_{ex}(0, y) = 1$')
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(R_values)))
    for idx, R_val in enumerate(R_values):
        y_p, u_p = profiles[R_val]
        ax_prof.plot(u_p, y_p, '--', color=colors[idx], linewidth=1.6, label=f'$R = 10^{{{int(math.log10(R_val))}}}$')

    ax_prof.set_xlabel("Horizontal Velocity $u_x(0, y)$", fontsize=11)
    ax_prof.set_ylabel("Channel Height $y$", fontsize=11)
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_prof = os.path.join(output_dir, "strategy_A_interface_profile.png")
    plt.savefig(plot_prof, dpi=300)
    plt.close(fig_prof)
    print(f">> Saved Strategy A Interface Profile Plot: {plot_prof}")


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_r_sweep_analysis(
        R_values=[1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7],
        fixed_n=60,
        Lx=4.0,
        Ly=1.0,
        L_buf=1.0,
        Re=40.0,
        T_end=5.0,
        dt=0.2,
        output_dir="results_buffer_recovery_v2"
    )