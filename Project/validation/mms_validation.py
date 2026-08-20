"""
Method of Manufactured Solutions (MMS) Validation Script.

Validates spatial convergence by invoking the exact project solver classes:
  - Conforming_solver (from Solvers.NS_Conforming)
  - Brinkman_solver (from Solvers.NS_Brinkman)
  - NS_DLM_Solver (from Solvers.NS_DLM_simple)

against a known, exact analytical solution:
    u_exact = [ sin(pi*x/Lx) * cos(pi*y/Ly), -(Ly/Lx) * cos(pi*x/Lx) * sin(pi*y/Ly) ]
    p_exact = sin(pi*x/Lx) * sin(pi*y/Ly)

Calculates exact L2/H1 velocity errors and L2 pressure errors directly
against the analytical field without grid interpolation error.
"""

import os
import sys
import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from firedrake import (
    Constant, SpatialCoordinate, VectorFunctionSpace, FunctionSpace,
    Function, CheckpointFile, assemble, dx, inner, grad, sin, cos, pi,
    as_vector, sqrt, div, nabla_grad, dot
)

# Ensure Project directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, "domain_settings"))
    sys.path.append(os.path.join(project_dir, "Utils"))
    sys.path.append(os.path.join(project_dir, "Solvers"))

from domain_settings.obstacles import squareObstacle, circleObstacle
from user_inputs.user_parameters import x_obs, y_obs, side_length, r_obs, Lx, Ly

from Solvers.NS_Conforming import Conforming_solver
from Solvers.NS_Brinkman import Brinkman_solver
from Solvers.NS_DLM_simple import NS_DLM_Solver


# =============================================================================
# 1. MMS EXACT SOLUTION DEFINITION
# =============================================================================

try:
    from firedrake import conditional as Conditional, conditional
except ImportError:
    from firedrake import conditional
    Conditional = conditional
from firedrake import sym, Identity, FacetNormal


def get_mms_exact_fields(mesh, Re=40.0, Lx=Lx, Ly=Ly, x_obs=x_obs, y_obs=y_obs, side_length=side_length, rho=1.0, L_char=0.2, u_char=1.0):
    """
    Costruisce u_exact, p_exact, f_mms e g_neumann per la validazione MMS con ostacolo quadrato.
    Garantisce:
      - div(u_exact) = 0
      - u_exact = 0 all'interno dell'ostacolo quadrato
      - u_exact = 0 sulle pareti y = 0 e y = Ly (No-Slip)
      - Flusso non nullo con u_y = 0 all'Inflow (x = 0)
    """
    X = SpatialCoordinate(mesh)
    x, y = X[0], X[1]

    # 1. Distanze esterne al quadrato (evitando sqrt per non dividere per zero)
    dx_val = Conditional(abs(x - x_obs) - side_length / 2.0 > 0.0, abs(x - x_obs) - side_length / 2.0, 0.0)
    dy_val = Conditional(abs(y - y_obs) - side_length / 2.0 > 0.0, abs(y - y_obs) - side_length / 2.0, 0.0)
    
    # r2 = r^2, quindi r4 = (r^2)^2 (puramente polinomiale, C^3 e senza 1/sqrt(0))
    r2 = dx_val**2 + dy_val**2
    r4 = r2 * r2

    # 2. Funzione di corrente psi
    psi = r4 * (sin(pi * y / Ly)**2) * cos(pi * x / Lx)

    # 3. Campo di velocita' esatto: div(u) = 0
    grad_psi = grad(psi)
    u_exact = as_vector([grad_psi[1], -grad_psi[0]])

    # 4. Pressione esatta (r2 * sqrt(r2) evitato: usiamo r2 per continuita' C^1)
    p_exact = r2 * cos(pi * x / Lx) * sin(pi * y / Ly)

    # 5. Parametri fisici
    mu = rho * L_char * u_char / Re

    # 6. Forzante MMS
    f_mms = rho * dot(u_exact, nabla_grad(u_exact)) - div(mu * sym(grad(u_exact))) + grad(p_exact)

    # 7. Flusso di trazione Neumann
    n_vec = FacetNormal(mesh)
    dim = mesh.geometric_dimension if isinstance(mesh.geometric_dimension, int) else (mesh.geometric_dimension() if callable(mesh.geometric_dimension) else 2)
    sigma_exact = mu * sym(grad(u_exact)) - p_exact * Identity(dim)
    g_neumann = dot(sigma_exact, n_vec)

    return u_exact, p_exact, f_mms, g_neumann


def get_case_directory(solver_name: str, obstacle: str, Re: float, n: int, R_penalty: float = 100000.0) -> str:
    """Returns the output directory path where solver checkpoints are saved for MMS."""
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    param_str = f"n{n}_R{R_penalty}_Re{Re}" if solver_name.lower() == "brinkman" else f"n{n}_Re{Re}"
    return os.path.join(project_dir, "Plots", "MMS", solver_name, "fixed", obstacle, sym_str, param_str)


def load_checkpoint_solutions(vel_file: str, pres_file: str):
    """Loads velocity and pressure functions onto the exact same mesh instance."""
    if not vel_file.endswith(".h5"):
        vel_file += ".h5"
    if not pres_file.endswith(".h5"):
        pres_file += ".h5"

    with CheckpointFile(vel_file, 'r') as chk_v:
        mesh = chk_v.load_mesh()
        uh = chk_v.load_function(mesh, "velocity")

    with CheckpointFile(pres_file, 'r') as chk_p:
        ph = chk_p.load_function(mesh, "pressure")

    return mesh, uh, ph


# =============================================================================
# 2. ERROR COMPUTATION AGAINST EXACT ANALYTICAL SOLUTION
# =============================================================================

def compute_exact_errors(mesh, uh, ph, u_exact, p_exact):
    """Computes exact L2(u), H1(u), and L2(p) error norms without grid interpolation."""
    err_u = uh - u_exact
    err_L2_u = sqrt(assemble(inner(err_u, err_u) * dx(domain=mesh)))
    err_H1_u = sqrt(assemble((inner(err_u, err_u) + inner(grad(err_u), grad(err_u))) * dx(domain=mesh)))

    vol = assemble(Constant(1.0) * dx(domain=mesh))
    mean_ph = assemble(ph * dx(domain=mesh)) / vol
    mean_pex = assemble(p_exact * dx(domain=mesh)) / vol

    err_p = (ph - mean_ph) - (p_exact - mean_pex)
    err_L2_p = sqrt(assemble(inner(err_p, err_p) * dx(domain=mesh)))

    return float(err_L2_u), float(err_H1_u), float(err_L2_p)


# =============================================================================
# 3. MAIN CONVERGENCE STUDY USING PROJECT SOLVER CLASSES
# =============================================================================

def run_mms_convergence(
    resolutions: List[int],
    obstacle_type: str = "square",
    solver_type: str = "conforming",
    Re: float = 40.0,
    R_penalty: float = 100000.0,
    scale_R: bool = False,
    dt: float = 0.5,
    t_final: float = 20.0
):
    dx_h = [1.0 / n for n in resolutions]
    errs_L2_u, errs_H1_u, errs_L2_p = [], [], []

    print("\n" + "=" * 70)
    print(f" STARTING MMS VALIDATION FOR {solver_type.upper()} CLASS ({obstacle_type.upper()})")
    print(f" Refinements n: {resolutions}")
    print(f" Reynolds number Re: {Re}")
    print(f" Time step dt: {dt}, t_final: {t_final}")
    if solver_type.lower() == "brinkman":
        print(f" Brinkman Resistance R: {R_penalty} (scaling with n: {scale_R})")
    print("=" * 70)

    n_min = resolutions[0]

    for n in resolutions:
        print(f"\n>> Instantiating and running {solver_type} solver class for n = {n} ...")

        u_exact_func = lambda m: get_mms_exact_fields(m, Re=Re)[0]
        p_exact_func = lambda m: get_mms_exact_fields(m, Re=Re)[1]
        f_mms_func = lambda m: get_mms_exact_fields(m, Re=Re)[2]
        g_neumann_func = lambda m: get_mms_exact_fields(m, Re=Re)[3]

        if solver_type.lower() == "conforming":
            solver = Conforming_solver(moving=False, type_obstacle=obstacle_type, n=n, Re=Re)
            solver.conforming_solve(f_custom=f_mms_func, u_exact=u_exact_func, p_exact=p_exact_func, g_custom=g_neumann_func, dt=dt, t_final=t_final)
            case_dir = get_case_directory("Conforming", obstacle_type, Re, n)

        elif solver_type.lower() == "brinkman":
            R_val = R_penalty * ((n / float(n_min)) ** 2) if scale_R else R_penalty
            solver = Brinkman_solver(moving=False, type_obstacle=obstacle_type, n=n, R=R_val, Re=Re)
            solver.Brinkman_solve(f_custom=f_mms_func, u_exact=u_exact_func, p_exact=p_exact_func, g_custom=g_neumann_func, dt=dt, t_final=t_final)
            case_dir = get_case_directory("Brinkman", obstacle_type, Re, n, R_val)

        elif solver_type.lower() in ("dlm", "ns_dlm"):
            solver = NS_DLM_Solver(moving=False, type_obstacle=obstacle_type, n=n, Re=Re)
            solver.NS_DLM_Solve(f_custom=f_mms_func, u_exact=u_exact_func, p_exact=p_exact_func, g_custom=g_neumann_func, dt=dt, t_final=t_final)
            case_dir = get_case_directory("DLM", obstacle_type, Re, n)

        else:
            raise ValueError(f"Unsupported solver: {solver_type}")

        # Load checkpoint solution produced by the solver class on the EXACT SAME mesh
        vel_file = os.path.join(case_dir, "velocity", f"velocity_t={t_final:.2f}.h5")
        pres_file = os.path.join(case_dir, "pressure", f"pressure_t={t_final:.2f}.h5")

        mesh_h, uh, ph = load_checkpoint_solutions(vel_file, pres_file)

        u_ex, p_ex, _, _ = get_mms_exact_fields(mesh_h, Re=Re)

        e_L2_u, e_H1_u, e_L2_p = compute_exact_errors(mesh_h, uh, ph, u_ex, p_ex)
        errs_L2_u.append(e_L2_u)
        errs_H1_u.append(e_H1_u)
        errs_L2_p.append(e_L2_p)

        print(f"   n = {n:3d} | h = {1.0/n:.4f} | L2(u) = {e_L2_u:.5e} | H1(u) = {e_H1_u:.5e} | L2(p) = {e_L2_p:.5e}")

    rates_L2_u, rates_H1_u, rates_L2_p = [], [], []
    for i in range(len(resolutions) - 1):
        log_h = np.log(dx_h[i] / dx_h[i + 1])
        rates_L2_u.append(np.log(errs_L2_u[i] / errs_L2_u[i + 1]) / log_h)
        rates_H1_u.append(np.log(errs_H1_u[i] / errs_H1_u[i + 1]) / log_h)
        rates_L2_p.append(np.log(errs_L2_p[i] / errs_L2_p[i + 1]) / log_h)

    print("\n" + "=" * 70)
    print(f" CONVERGENCE RATES SUMMARY FOR CLASS {solver_type.upper()}")
    print("=" * 70)
    print(f"{'n':>5} | {'h':>8} | {'L2(u) Error':>14} | {'H1(u) Error':>14} | {'L2(p) Error':>14}")
    print("-" * 70)
    for i, n in enumerate(resolutions):
        print(f"{n:5d} | {dx_h[i]:8.4f} | {errs_L2_u[i]:14.5e} | {errs_H1_u[i]:14.5e} | {errs_L2_p[i]:14.5e}")
    print("-" * 70)
    print("Empirical Convergence Rates:")
    for i in range(len(resolutions) - 1):
        print(f"  n = {resolutions[i]} -> {resolutions[i+1]}: "
              f"Rate L2(u) = {rates_L2_u[i]:+.3f} | "
              f"Rate H1(u) = {rates_H1_u[i]:+.3f} | "
              f"Rate L2(p) = {rates_L2_p[i]:+.3f}")
    print("=" * 70 + "\n")

    # Plot
    case_name = "Brinkman" if solver_type.lower() == "brinkman" else ("DLM" if solver_type.lower() in ("dlm", "ns_dlm") else "Conforming")
    out_dir = os.path.join(project_dir, "Plots", "MMS")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"mms_convergence_{case_name}_{obstacle_type}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    h_arr = np.array(dx_h)
    ax.loglog(h_arr, errs_L2_u, 'o-', color='#2c7bb6', linewidth=2, label='$L^2$ Velocity Error')
    ax.loglog(h_arr, errs_H1_u, 's-', color='#d7191c', linewidth=2, label='$H^1$ Velocity Error')
    ax.loglog(h_arr, errs_L2_p, '^--', color='#2b83ba', linewidth=1.8, label='$L^2$ Pressure Error')

    ref_1 = errs_L2_u[0] / h_arr[0]
    ref_2 = errs_L2_u[0] / (h_arr[0]**2)
    ax.loglog(h_arr, ref_1 * h_arr, 'k--', alpha=0.5, label='$O(h)$')
    ax.loglog(h_arr, ref_2 * (h_arr**2), 'k:', alpha=0.5, label='$O(h^2)$')

    ax.set_xlabel("Mesh size $h = 1/n$")
    ax.set_ylabel("Error Norm vs Exact Analytical Solution")
    ax.set_title(f"MMS Exact Solution Convergence — {case_name} ({obstacle_type.capitalize()})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f">> Saved MMS plot to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    # =========================================================================
    # EDIT MMS VALIDATION STUDY PARAMETERS HERE
    # =========================================================================
    resolutions = [50, 75, 100, 150]                     # Mesh refinement levels n to simulate
    obstacle_type = "square"                            # "square" or "cylinder"
    solvers_to_test = ["brinkman"]                      # List of solvers to include in MMS study
    #solvers_to_test = ["conforming", "brinkman", "dlm"] # List of solvers to include in MMS study
    Re = 40.0                                           # Reynolds number
    R_penalty = 10000.0                                # Resistive parameter R (for Brinkman solver)
    scale_R = False                                     # Scale Brinkman resistance R(n) = R_0 * (n/n_min)^2
    dt = 0.01                                        # Time step dt for time integration
    t_final = 0.05                                   # Final simulation time step t_final
    # =========================================================================


    for s_type in solvers_to_test:
        run_mms_convergence(
            resolutions=resolutions,
            obstacle_type=obstacle_type,
            solver_type=s_type,
            Re=Re,
            R_penalty=R_penalty,
            scale_R=scale_R,
            dt=dt,
            t_final=t_final
        )
