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

from domain_settings.obstacles import BufferObstacle
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
    nx_phys = n
    nx_buf = max(1, int(round(n * L_buf / Lx)))
    n_tot = nx_buf + nx_phys
    ny = max(4, int(round(n * Ly / Lx)))
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
    h_vals = [Lx / n for n in resolutions]

    for n in resolutions:
        print(f"\n---> Running Resolution n = {n:3d} (h = {Lx/n:.4f}) ...")

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

    # -------------------------------------------------------------------------
    # 5. PRINT CONVERGENCE SUMMARY TABLES (via experiment_plots module)
    # -------------------------------------------------------------------------
    from experiment_plots import (
        print_phase_comparison_tables,
        plot_phase_comparison_loglog,
        plot_interface_velocity_profile
    )

    print_phase_comparison_tables(
        resolutions=resolutions,
        h_vals=h_vals,
        res_p1=res_p1,
        res_p2=res_p2,
        method_name="DLM"
    )

    # -------------------------------------------------------------------------
    # 6. GENERATE COMPARISON PLOTS (via experiment_plots module)
    # -------------------------------------------------------------------------

    # Plot 1: Spatial Log-Log Convergence Plot
    conv_plot_path = os.path.join(output_dir, "convergence_comparison_loglog.png")
    plot_phase_comparison_loglog(
        h_vals=h_vals,
        res_p1=res_p1,
        res_p2=res_p2,
        method_name="DLM",
        output_path=conv_plot_path
    )

    # Plot 2: Interface Velocity Cut Profile
    prof_plot_path = os.path.join(output_dir, "interface_velocity_recovery.png")
    dlm_labels = [f"DLM Rec. (n={n})" for n in resolutions]
    plot_interface_velocity_profile(
        profiles=profiles_p2,
        Ly=Ly,
        keys=resolutions,
        title="Velocity Profile Recovery at Interface $\\Sigma$ ($x = 0$)",
        output_path=prof_plot_path,
        custom_labels=dlm_labels,
        xlim=(0.8, 1.2)
    )


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