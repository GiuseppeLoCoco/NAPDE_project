"""
Numerical Experiment: Upstream Buffer Layer Dirichlet Recovery via Distributed Lagrange Multipliers (DLM)
"""

import os
import sys
import gc
import math
import warnings
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Compatibility for NumPy 1.x and 2.x trapezoidal integration
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from firedrake import (
    RectangleMesh, VectorFunctionSpace, FunctionSpace, Function,
    TrialFunction, TestFunction, TrialFunctions, TestFunctions,
    DirichletBC, Constant, SpatialCoordinate, as_vector,
    inner, dot, grad, sym, div, nabla_grad, dx, ds, sqrt,
    assemble, solve, conditional, lt, ge, sin, cos, pi
)

from firedrake import MixedVectorSpaceBasis, VectorSpaceBasis


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
        self.L_char = 0.2  # Characteristic obstacle/channel scale
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
        """Analytical momentum source term: f = rho*(u.grad)u - div(2*mu*sym(grad(u))) + grad(p)."""
        u_ex = self.u_exact(X)
        p_ex = self.p_exact(X)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press


# =============================================================================
# 2. FRACTIONAL-STEP DLM BUFFER SOLVER
# =============================================================================

def solve_dlm_buffer(n: int, mms: ManufacturedSolution,
                     Lx: float = 4.0, Ly: float = 1.0, L_buf: float = 1.0,
                     T_end: float = 2.0, dt: float = 0.5) -> Tuple[Function, Function, Function, object]:
    """
    Solves Navier-Stokes on the extended domain [-L_buf, Lx] x [0, Ly] using the DLM fractional-step method.
    The buffer region [-L_buf, 0) is enforced to match u_exact via a distributed Lagrange multiplier field.
    """
    # Grid node alignment: ensures the interface x = 0 lies exactly on mesh cell vertices
    nx_buf = int(round(L_buf * n))
    nx_phys = int(round(Lx * n))
    n_tot = nx_buf + nx_phys
    ny = int(round(Ly * n))
    L_tot = L_buf + Lx

    mesh = RectangleMesh(n_tot, ny, L_tot, Ly)
    mesh.coordinates.dat.data[:, 0] -= L_buf

    X = SpatialCoordinate(mesh)
    x = X[0]

    # Taylor-Hood spaces (P2-P1) for fluid; P1 vector space for Lagrange Multiplier
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    Z = VectorFunctionSpace(mesh, "CG", 1)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    u_ex = mms.u_exact(X)
    p_ex = mms.p_exact(X)
    f_val = mms.f_forcing(X)

    # Indicator function: chi_buf = 1 inside buffer (x < 0), 0 inside physical domain (x >= 0)
    chi_buf = conditional(lt(x, 0.0), 1.0, 0.0)

    # Boundary conditions for tentative velocity: Dirichlet on top/bottom walls (1, 2) and outlet (4).
    # Upstream inlet (3, x = -L_buf) has no Dirichlet constraint (free natural Neumann traction).
    bcs_tentative = [
        DirichletBC(W.sub(0), u_ex, 2),
        DirichletBC(W.sub(0), u_ex, 3),
        DirichletBC(W.sub(0), u_ex, 4)
    ]

    # Boundary conditions for velocity correction step
    bcs_correction = [
        DirichletBC(V, u_ex, 2),
        DirichletBC(V, u_ex, 3),
        DirichletBC(V, u_ex, 4)
    ]

    # -------------------------------------------------------------------------
    # WARM START: Inizialization with Stokes (Linearity)
    # -------------------------------------------------------------------------
    uh_n = Function(V)
    sol_init = Function(W)
    
    a_init = 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
                - div(v) * p * dx \
                + div(u) * q * dx
    
    L_init = inner(f_val, v) * dx
    
    solve(a_init == L_init, sol_init, bcs=bcs_tentative, 
            solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

    uh_n.assign(sol_init.subfunctions[0])
    # -------------------------------------------------------------------------

    uh_n_ex = Function(V)
    # uh_n.assign(0.0)
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

    lambda_n = Function(Z)
    lambda_n.assign(0.0)

    lambda_new = Function(Z)
    sol_star = Function(W)
    u_star, ph = sol_star.subfunctions
    uh = Function(V)

    # -------------------------------------------------------------------------
    # STEP 1: Tentative Navier-Stokes Problem (with previous multiplier forcing)
    # -------------------------------------------------------------------------
    a1 = (Constant(mms.rho) / Constant(dt)) * inner(u, v) * dx \
        + Constant(mms.rho) * inner(dot(uh_n, nabla_grad(u)), v) * dx \
        + 2.0 * Constant(mms.mu) * inner(sym(grad(u)), sym(grad(v))) * dx \
        - div(v) * p * dx \
        + div(u) * q * dx

    # Without the stabilization term:
    # " Constant(mms.rho)*div(uh_n)*inner(u, v)*dx "

    L1 = (Constant(mms.rho) / Constant(dt)) * inner(uh_n, v) * dx \
        + inner(f_val, v) * dx \
        - inner(chi_buf * lambda_n, v) * dx

    # -------------------------------------------------------------------------
    # STEP 2: Distributed Lagrange Multiplier Update
    # -------------------------------------------------------------------------
    lam = TrialFunction(Z)
    e = TestFunction(Z)

    a2 = inner(lam, e) * dx
    L2 = inner(chi_buf * lambda_n, e) * dx \
        + (Constant(mms.rho) / Constant(dt)) * inner(chi_buf * (u_star - u_ex), e) * dx

    # -------------------------------------------------------------------------
    # STEP 3: Velocity Correction Step
    # -------------------------------------------------------------------------
    u_corr = TrialFunction(V)
    v_corr = TestFunction(V)

    a3 = (Constant(mms.rho) / Constant(dt)) * inner(u_corr, v_corr) * dx
    L3 = (Constant(mms.rho) / Constant(dt)) * inner(u_star, v_corr) * dx \
        - inner(chi_buf * (lambda_new - lambda_n), v_corr) * dx

    # Solver parameter configurations
    solver_lu_mumps = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_mumps_icntl_14': 80
    }
    solver_cg_sor = {
        'ksp_type': 'cg',
        'pc_type': 'sor'
    }

    num_steps = max(1, int(round(T_end / dt)))

    num_steps_max = 300   
    tol_steady = 1e-7  
    t_val = 0.0

    for step in range(num_steps_max):
        t_val += dt

        # Step 1: Solve tentative velocity and pressure
        solve(a1 == L1, sol_star, bcs=bcs_tentative, solver_parameters=solver_lu_mumps,
              form_compiler_parameters={'quadrature_degree': 8})

        # Step 2: Solve Lagrange multiplier on buffer region
        solve(a2 == L2, lambda_new, solver_parameters=solver_cg_sor,
              form_compiler_parameters={'quadrature_degree': 8})

        # Step 3: Solve velocity projection / correction (cut the BCs)
        solve(a3 == L3, uh, solver_parameters=solver_cg_sor,
              form_compiler_parameters={'quadrature_degree': 8})

        diff_u = uh - uh_n
        # increment_L2 = sqrt(assemble(inner(diff_u, diff_u) * dx(domain=mesh)))
        increment_H1 = sqrt(assemble((inner(diff_u, diff_u) + inner(grad(diff_u), grad(diff_u))) * dx(domain=mesh)))

        uh_n.assign(uh)
        lambda_n.assign(lambda_new)

        if float(increment_H1) < tol_steady:
            print(f"      [!] Steady-state reach at step {step + 1} (t = {t_val:.2f}) | Increment: {float(increment_H1):.2e}")
            break
    else:
        print(f"      [!!] WARNING: steady state NOT reached after {num_steps_max} steps "
            f"(last increment: {float(increment_H1):.2e})")

    # Memory cleanup of temporary UFL variational forms
    del a1, L1, a2, L2, a3, L3, bcs_tentative, bcs_correction
    gc.collect()

    return uh, ph, lambda_n, mesh


# =============================================================================
# 3. ERROR EVALUATION & PROFILE EXTRACTION
# =============================================================================

def compute_restricted_errors(mesh, uh, ph, mms: ManufacturedSolution) -> Tuple[float, float, float]:
    """Computes velocity L2, H1 and pressure L2 errors restricted strictly to Omega_0 (x >= 0)."""
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
# 4. EXPERIMENTAL CONVERGENCE PIPELINE
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
    print("UPSTREAM BUFFER RECOVERY EXPERIMENT: DISTRIBUTED LAGRANGE MULTIPLIERS (DLM)")
    print(f"Domain: Physical [0, {Lx}] x [0, {Ly}] + Buffer [-{L_buf}, 0] x [0, {Ly}] | Re = {Re}")
    print(f"Mesh Resolutions n: {resolutions} | Final Time T: {T_end}s (dt = {dt}s)")
    print("=" * 90)

    errs_L2_u, errs_H1_u, errs_L2_p, errs_intf = [], [], [], []
    profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    h_vals = [1.0 / n for n in resolutions]

    for n in resolutions:

        h = 1.0 / n
        # dt_scaled = 0.5 * (h / (1.0/resolutions[0]))**2  # Scaled with O(h^2)
        # print(f"\n---> Running DLM Resolution n = {n:3d} (h = {h:.4f}, dt = {dt_scaled:.2e}) ...")

        # uh, ph, lambda_h, mesh = solve_dlm_buffer(n, mms, Lx, Ly, L_buf, T_end, dt_scaled)

        print(f"\n---> Running DLM Resolution n = {n:3d} (h = {1.0/n:.4f}) ...")
        uh, ph, lambda_h, mesh = solve_dlm_buffer(n, mms, Lx, Ly, L_buf, T_end, dt)

        e_L2, e_H1, e_p = compute_restricted_errors(mesh, uh, ph, mms)
        y_pts, u_x_num, u_x_ex = extract_interface_profile(uh, mms)
        e_intf = float(np.sqrt(_trapezoid((u_x_num - u_x_ex)**2, y_pts)))

        errs_L2_u.append(e_L2)
        errs_H1_u.append(e_H1)
        errs_L2_p.append(e_p)
        errs_intf.append(e_intf)
        profiles[n] = (y_pts, u_x_num)

        print(f"     [DLM] L2(u) in Omega_0: {e_L2:.5e} | H1(u): {e_H1:.5e} | L2(p): {e_p:.5e} | Interface L2: {e_intf:.5e}")

        # Explicit cleanup per iteration
        del uh, ph, lambda_h, mesh
        gc.collect()

    # Compute empirical spatial convergence rates
    def compute_rates(err_list):
        return [np.log(err_list[i] / err_list[i+1]) / np.log(h_vals[i] / h_vals[i+1]) for i in range(len(h_vals)-1)]

    rates_L2 = compute_rates(errs_L2_u)
    rates_H1 = compute_rates(errs_H1_u)
    rates_intf = compute_rates(errs_intf)

    # Print summary table
    print("\n" + "=" * 105)
    print("CONVERGENCE SUMMARY TABLE: DISTRIBUTED LAGRANGE MULTIPLIERS (DLM) BUFFER RECOVERY")
    print("=" * 105)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Err (Omega_0)':>20} | {'Rate':>6} | {'H1(u) Err':>12} | {'Rate':>6} | {'Interface L2':>14} | {'Rate':>6}")
    print("-" * 105)
    for i, n in enumerate(resolutions):
        r_l2_str = f"{rates_L2[i-1]:+6.2f}" if i > 0 else "    --"
        r_h1_str = f"{rates_H1[i-1]:+6.2f}" if i > 0 else "    --"
        r_intf_str = f"{rates_intf[i-1]:+6.2f}" if i > 0 else "    --"
        print(f"{n:5d} | {h_vals[i]:8.4f} | {errs_L2_u[i]:20.5e} | {r_l2_str} | {errs_H1_u[i]:12.5e} | {r_h1_str} | {errs_intf[i]:14.5e} | {r_intf_str}")
    print("=" * 105 + "\n")

    # =========================================================================
    # 5. GENERATE COMPARISON PLOTS
    # =========================================================================

    # Plot 1: Spatial Log-Log Convergence Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Spatial Convergence: Upstream Buffer Recovery via DLM", fontsize=13, fontweight='bold')

    h_arr = np.array(h_vals)
    ax1.loglog(h_arr, errs_L2_u, 'o-', color='#1f77b4', linewidth=2, label='Velocity $L^2(\\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_H1_u, 's--', color='#ff7f0e', linewidth=1.8, label='Velocity $H^1(\\Omega_0)$ Error')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label='Optimal $O(h^2)$')
    ax1.loglog(h_arr, errs_L2_u[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label='Slope $O(h)$')
    ax1.set_xlabel("Mesh Size $h = 1/n$", fontsize=11)
    ax1.set_ylabel("Velocity Error Norms in $\\Omega_0$", fontsize=11)
    ax1.set_title("Restricted Velocity Error in Physical Domain", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9.5)

    ax2.loglog(h_arr, errs_intf, 'd-', color='#d62728', linewidth=2, label='Interface Trace $L^2(\\Sigma)$ Error')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**2, 'k:', alpha=0.6, label='Optimal $O(h^2)$')
    ax2.loglog(h_arr, errs_intf[0] * (h_arr / h_arr[0])**1, 'k--', alpha=0.6, label='Slope $O(h)$')
    ax2.set_xlabel("Mesh Size $h = 1/n$", fontsize=11)
    ax2.set_ylabel("Trace Error at Interface $\\Sigma$ ($x = 0$)", fontsize=11)
    ax2.set_title("Dirichlet Interface Recovery vs $h$", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9.5)

    plt.tight_layout()
    conv_plot_path = os.path.join(output_dir, "dlm_spatial_convergence.png")
    plt.savefig(conv_plot_path, dpi=300)
    plt.close(fig)
    print(f">> Saved DLM Convergence Plot: {conv_plot_path}")

    # Plot 2: Interface Velocity Cut Profile
    fig_prof, ax_prof = plt.subplots(figsize=(7, 6))
    fig_prof.suptitle("Interface Velocity Profile $u_x(0, y)$ Recovery via DLM", fontsize=12, fontweight='bold')

    y_fine = np.linspace(0, Ly, 200)
    ax_prof.plot(np.zeros_like(y_fine), y_fine, 'k-', linewidth=2.5, label='Target $\\mathbf{u}_{ex}(0, y) = 0$')

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(resolutions)))
    for idx, n in enumerate(resolutions):
        y_p, u_p = profiles[n]
        ax_prof.plot(u_p, y_p, '--', color=colors[idx], linewidth=1.6, label=f'DLM ($n = {n}$)')

    ax_prof.set_xlabel("Horizontal Velocity $u_x(0, y)$", fontsize=11)
    ax_prof.set_ylabel("Channel Height $y$", fontsize=11)
    ax_prof.grid(True, linestyle="--", alpha=0.5)
    ax_prof.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    prof_plot_path = os.path.join(output_dir, "dlm_interface_profile.png")
    plt.savefig(prof_plot_path, dpi=300)
    plt.close(fig_prof)
    print(f">> Saved DLM Interface Profile Plot: {prof_plot_path}")


# =============================================================================
# 5. ENTRY POINT
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