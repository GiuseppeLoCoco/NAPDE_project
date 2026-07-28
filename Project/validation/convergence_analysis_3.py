"""
Convergence analysis pipeline for penalization methods applied to flow past
an obstacle.

This module was rewritten from scratch to be fully configurable. Instead of
being hard-wired to a single method / single obstacle / single Reynolds
number, the analysis to run is now described by an ``AnalysisConfig`` object
(see the ``CONFIG`` section and the ``__main__`` block at the bottom of the
file). From that configuration the code automatically:

    1. builds every relevant combination of
           method      in {"Brinkman", "DLM"}
           obstacle    in {"cylinder", "square"}
           reynolds    in {40, 80}
           symmetry    in {"symmetric", "asymmetric"}   (cylinder only)
    2. resolves the corresponding directory tree, e.g.:

           Brinkman/fixed/square/steady/n150_R1000.0_Re40
           DLM/moving/cylinder/unsteady/symmetric/n100_Re80

    3. loads the reference ("conforming") solution and every penalized
       solution, computes the L2/H1 fluid-only errors (and pressure error),
       and produces the usual convergence plots, profile plots and summary
       files -- one full set of outputs per combination.

Directory-naming rules
-----------------------
- ``motion``  is fixed by the obstacle: "moving" for the cylinder, "fixed"
  for the square (see ``OBSTACLE_MOTION``).
- ``regime``  is fixed by the Reynolds number: Re = 40 -> "steady",
  Re = 80 -> "unsteady" (see ``REYNOLDS_REGIME``).
- ``symmetry`` (a "symmetric"/"asymmetric" sub-folder) is present for BOTH
  obstacles, but only in the unsteady regime (Re = 80); the steady cases
  (Re = 40) never have this folder level, for either obstacle. This is
  controlled by ``OBSTACLE_HAS_SYMMETRY`` + ``case_has_symmetry``.
- The penalization parameter R only appears in the folder/file name for the
  Brinkman method (kept fixed at ``R_penalty``, default 1000.0); DLM folders
  have no R.

NOTE ON THE REFERENCE ("CONFORMING") SOLUTION
----------------------------------------------
The reference/conforming solution (n = 200, taken as the "exact" solution)
lives under its own tree, with NO symmetric/asymmetric sub-folder (even for
the cylinder), and with the square spelled "squared" at this level only:

    Conforming/fixed/square/steady/n200_Re40
    Conforming/moving/cylinder/unsteady/n200_Re80

This is handled by ``CONFORMING_OBSTACLE_NAME`` and
``PathBuilder.reference_case_dir``. If any of this changes, only those two
spots need editing; nothing else in the pipeline depends on it.
"""

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from firedrake import (Constant, project, assemble, dx, inner, grad, sqrt,
                        CheckpointFile)


# =============================================================================
# 0. STRUCTURAL RULES (obstacle <-> motion, Reynolds <-> regime, symmetry)
# =============================================================================

# Motion regime is entirely determined by the obstacle type.
OBSTACLE_MOTION = {
    "cylinder": "moving",
    "square":   "fixed",
}

# Time regime is entirely determined by the Reynolds number.
REYNOLDS_REGIME = {
    40: "steady",
    80: "unsteady",
}

# Whether the directory tree for a given obstacle includes a
# "symmetric" / "asymmetric" sub-folder AT ALL (structural capability).
# Both obstacles have this split (in the unsteady regime only -- see
# case_has_symmetry below).
OBSTACLE_HAS_SYMMETRY = {
    "cylinder": True,
    "square":   True,
}


def case_has_symmetry(obstacle: str, Re: int) -> bool:
    """
    Whether THIS specific (obstacle, Re) combination actually has a
    symmetric/asymmetric sub-folder. Currently: both obstacles, but only in
    the unsteady regime (Re = 80) -- the steady cases (Re = 40) have no
    symmetry split, for either obstacle.
    """
    regime = REYNOLDS_REGIME[Re]
    return OBSTACLE_HAS_SYMMETRY.get(obstacle, False) and regime == "unsteady"

# Obstacle folder name as it appears specifically under the "Conforming/"
# tree. Confirmed layout:
#   Conforming/fixed/square/steady/n200_Re40
#   Conforming/moving/cylinder/unsteady/n200_Re80
CONFORMING_OBSTACLE_NAME = {
    "cylinder": "cylinder",
    "square":   "square",
}

VALID_SYMMETRIES = ("symmetric", "asymmetric")
VALID_METHODS = ("Brinkman", "DLM")
VALID_OBSTACLES = ("cylinder", "square")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Top-level configuration for the whole convergence-analysis campaign.
    Edit the defaults here (or override them when instantiating the class)
    to select exactly which methods / obstacles / Reynolds numbers /
    symmetries and mesh resolutions should be analyzed.
    """

    # Root folder containing the "Brinkman/", "DLM/" (and "Conforming/") trees
    base_dir: str

    # Root folder where results (plots, csv, summaries) will be written
    output_dir: str = "results"

    # --- What to analyze -----------------------------------------------
    methods:     List[str] = field(default_factory=lambda: list(VALID_METHODS))
    obstacles:   List[str] = field(default_factory=lambda: list(VALID_OBSTACLES))
    reynolds:    List[int] = field(default_factory=lambda: [40, 80])
    symmetries:  List[str] = field(default_factory=lambda: list(VALID_SYMMETRIES))
    resolutions: List[int] = field(default_factory=lambda: [50, 100, 150])

    # --- Fixed numerical parameters --------------------------------------
    R_penalty: float = 1000.0   # Brinkman penalization parameter (not varied)
    reference_n: int = 200      # mesh resolution of the conforming reference

    # --- Time sampling ----------------------------------------------------
    # "steady" cases (Re = 40): a single snapshot is analyzed.
    t_steady: float = 10.0
    # "unsteady" cases (Re = 80): a full time history is analyzed (Bochner norms).
    t_start_unsteady: float = 0.5
    t_end_unsteady: float = 10.0
    dt_unsteady: float = 0.5

    Ly: float = 1.0  # vertical extent of the domain, for profile extraction

    def time_steps_for(self, regime: str) -> np.ndarray:
        if regime == "steady":
            return np.array([self.t_steady])
        elif regime == "unsteady":
            return np.arange(self.t_start_unsteady,
                              self.t_end_unsteady + self.dt_unsteady,
                              self.dt_unsteady)
        raise ValueError(f"Unknown regime '{regime}'")

    def final_time_for(self, regime: str) -> float:
        return self.time_steps_for(regime)[-1]


# =============================================================================
# 2. PATH BUILDING
# =============================================================================

class PathBuilder:
    """
    Resolves the directory / file-name conventions of the new layout.

        Brinkman/fixed/square/steady/n150_R1000.0_Re40/velocity/velocity_t=...h5
        DLM/moving/cylinder/unsteady/symmetric/n100_Re80/vtk/pressure/pressure_t=...h5

    NOTE: only DLM output has the extra 'vtk' sub-folder between the case
    folder and 'velocity'/'pressure'; Brinkman and Conforming do not.
    """

    def __init__(self, base_dir: str, cfg: AnalysisConfig):
        self.base_dir = base_dir
        self.cfg = cfg

    # --- penalized-solution paths ---------------------------------------

    def folder_name(self, method: str, n: int, Re: int) -> str:
        if method.lower() == "brinkman":
            return f"n{n}_R{self.cfg.R_penalty}_Re{Re}"
        elif method.lower() == "dlm":
            return f"n{n}_Re{Re}"
        raise ValueError(f"Unknown method '{method}'")

    def case_dir(self, method: str, obstacle: str, Re: int, n: int,
                 symmetry: Optional[str] = None) -> str:
        motion = OBSTACLE_MOTION[obstacle]
        regime = REYNOLDS_REGIME[Re]
        parts = [self.base_dir, method, motion, obstacle, regime]
        if case_has_symmetry(obstacle, Re):
            if symmetry is None:
                raise ValueError(
                    f"Case (obstacle='{obstacle}', Re={Re}) requires a "
                    f"symmetry value in {VALID_SYMMETRIES}."
                )
            parts.append(symmetry)
        parts.append(self.folder_name(method, n, Re))
        return os.path.join(*parts)

    def velocity_file(self, method, obstacle, Re, n, t, symmetry=None) -> str:
        case_dir = self.case_dir(method, obstacle, Re, n, symmetry)
        if method.lower() == "dlm":
            # DLM outputs have an extra 'vtk' layer: n{...}/vtk/velocity/...
            case_dir = os.path.join(case_dir, "vtk")
        d = os.path.join(case_dir, "velocity")
        return os.path.join(d, f"velocity_t={t:.2f}.h5")

    def pressure_file(self, method, obstacle, Re, n, t, symmetry=None) -> str:
        case_dir = self.case_dir(method, obstacle, Re, n, symmetry)
        if method.lower() == "dlm":
            # DLM outputs have an extra 'vtk' layer: n{...}/vtk/pressure/...
            case_dir = os.path.join(case_dir, "vtk")
        d = os.path.join(case_dir, "pressure")
        return os.path.join(d, f"pressure_t={t:.2f}.h5")

    # --- reference / conforming solution paths ---------------------------
    # See module docstring: this convention is an assumption and is the only
    # place that needs editing if the real layout differs.

    def reference_case_dir(self, obstacle: str, Re: int,
                            symmetry: Optional[str] = None) -> str:
        # NOTE: the reference/conforming solution has NO symmetric/
        # asymmetric sub-folder, even for the cylinder -- `symmetry` is
        # accepted here only for a uniform call signature with the
        # penalized-solution path builders, and is intentionally ignored.
        motion = OBSTACLE_MOTION[obstacle]
        regime = REYNOLDS_REGIME[Re]
        conforming_obstacle_name = CONFORMING_OBSTACLE_NAME[obstacle]
        parts = [self.base_dir, "Conforming", motion, conforming_obstacle_name, regime]
        parts.append(f"n{self.cfg.reference_n}_Re{Re}")
        return os.path.join(*parts)

    def reference_velocity_file(self, obstacle, Re, t, symmetry=None) -> str:
        d = os.path.join(self.reference_case_dir(obstacle, Re, symmetry), "velocity")
        return os.path.join(d, f"velocity_t={t:.2f}.h5")

    def reference_pressure_file(self, obstacle, Re, t, symmetry=None) -> str:
        d = os.path.join(self.reference_case_dir(obstacle, Re, symmetry), "pressure")
        return os.path.join(d, f"pressure_t={t:.2f}.h5")


# =============================================================================
# 3. OBSTACLE-DEPENDENT HELPERS (profile extraction center / solid velocity)
# =============================================================================

def _cylinder_center_and_velocity(obstacle, t: float) -> Tuple[float, float]:
    """Moving-cylinder centre x-coordinate and prescribed rigid velocity."""
    omega = 0.2 * math.pi
    displ_x = obstacle.amplitude * 0.5 * (1.0 - math.cos(omega * t))
    x_c = obstacle.x_obs + displ_x
    v_solid_x = obstacle.amplitude * 0.5 * omega * math.sin(omega * t)
    return x_c, v_solid_x


def _square_center_and_velocity(obstacle, t: float) -> Tuple[float, float]:
    """Fixed square: centre never moves, solid velocity is zero."""
    return obstacle.x_obs, 0.0


CENTER_AND_VELOCITY_FN: Dict[str, Callable] = {
    "cylinder": _cylinder_center_and_velocity,
    "square":   _square_center_and_velocity,
}


# =============================================================================
# 4. POST-PROCESSING & ANALYSIS CLASS
# =============================================================================

class ConvergenceAnalyzer:
    """
    Handles error computation (fluid-only L2/H1, pressure L2) and vertical
    velocity-profile extraction, independent of the specific obstacle type.
    """

    def __init__(self, obstacle_instance, obstacle_kind: str, Ly: float = 1.0):
        """
        obstacle_instance : the physical obstacle object (must expose a
                             .chi(mesh, t) indicator function, plus whatever
                             attributes its CENTER_AND_VELOCITY_FN needs).
        obstacle_kind     : "cylinder" or "square" -- selects which
                             center/velocity convention to use for profiles.
        """
        self.obstacle = obstacle_instance
        self.obstacle_kind = obstacle_kind
        self.Ly = Ly
        self._center_fn = CENTER_AND_VELOCITY_FN[obstacle_kind]

    def compute_fluid_errors(self, u_ex, u_h, t_val):
        """L2 and H1 error norms restricted to the fluid domain."""
        V_ex = u_ex.function_space()
        mesh_ex = V_ex.mesh()
        t_const = Constant(t_val)

        u_h_proj = project(u_h, V_ex)

        chi_solid = self.obstacle.chi(mesh_ex, t_const)
        mask_fluid = 1.0 - chi_solid

        err = u_h_proj - u_ex

        err_L2_sq = assemble(mask_fluid * inner(err, err) * dx)
        err_H1_sq = assemble(mask_fluid * (inner(err, err) + inner(grad(err), grad(err))) * dx)

        return sqrt(err_L2_sq), sqrt(err_H1_sq)

    def compute_pressure_error_L2(self, p_ex, p_h, t_val):
        """L2 error norm for pressure, restricted to the fluid domain."""
        V_ex = p_ex.function_space()
        mesh_ex = V_ex.mesh()
        t_const = Constant(t_val)

        p_h_proj = project(p_h, V_ex)

        chi_solid = self.obstacle.chi(mesh_ex, t_const)
        mask_fluid = 1.0 - chi_solid

        vol_fluid = assemble(mask_fluid * dx(domain=mesh_ex))
        mean_p_ex = assemble(mask_fluid * p_ex * dx) / vol_fluid
        mean_p_h = assemble(mask_fluid * p_h_proj * dx) / vol_fluid

        err_p = (p_h_proj - mean_p_h) - (p_ex - mean_p_ex)
        err_p_L2_sq = assemble(mask_fluid * inner(err_p, err_p) * dx)
        return sqrt(err_p_L2_sq)

    def extract_vertical_profile(self, u_h, t_val, num_points=200):
        """
        Velocity-magnitude profile along the vertical line through the
        obstacle's (possibly moving) centre. Inside a conforming hole, the
        prescribed rigid-body velocity is returned instead of failing.
        """
        x_c, v_solid_x = self._center_fn(self.obstacle, t_val)

        y_coords = np.linspace(1e-5, self.Ly - 1e-5, num_points)
        u_mag = np.zeros_like(y_coords)

        for i, y in enumerate(y_coords):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    val = u_h.at([x_c, y], tolerance=1e-6)
                u_mag[i] = math.sqrt(val[0] ** 2 + val[1] ** 2)
            except Exception:
                # Point falls inside the conforming mesh's solid hole.
                u_mag[i] = abs(v_solid_x)

        return y_coords, u_mag


# =============================================================================
# 5. OUTPUT DIRECTORY SETUP & HDF5 LOADING
# =============================================================================

def setup_output_dirs(base_output_dir: str) -> Dict[str, str]:
    """
    Creates:
      <base_output_dir>/
        plots/profiles/
        plots/convergence/
        data/
    """
    dirs = {
        "root":       base_output_dir,
        "plots":      os.path.join(base_output_dir, "plots"),
        "profiles":   os.path.join(base_output_dir, "plots", "profiles"),
        "conv_plots": os.path.join(base_output_dir, "plots", "convergence"),
        "data":       os.path.join(base_output_dir, "data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f">> Output directory: {base_output_dir}")
    return dirs


def load_hdf5_solution(checkpoint_path: str, field_name: str = "velocity"):
    """Loads a saved Firedrake function from a CheckpointFile .h5 file."""
    if checkpoint_path.endswith(".h5"):
        checkpoint_path = checkpoint_path[:-3]
    checkpoint_path = checkpoint_path + ".h5"

    with CheckpointFile(checkpoint_path, 'r') as chk:
        mesh = chk.load_mesh()
        field = chk.load_function(mesh, field_name)

    return field


# =============================================================================
# 6. SINGLE-CASE PIPELINE (one method / obstacle / Re / symmetry combination)
# =============================================================================

def run_case(cfg: AnalysisConfig,
             method: str,
             obstacle: str,
             Re: int,
             symmetry: Optional[str],
             obstacle_instance,
             paths: PathBuilder):
    """
    Runs the full convergence analysis for a single
    (method, obstacle, Reynolds, symmetry) combination across all configured
    mesh resolutions, and writes plots/CSVs/summary under
    <output_dir>/<case_label>/.
    """
    regime = REYNOLDS_REGIME[Re]
    case_label = f"{method}_{obstacle}_Re{Re}"
    if symmetry is not None:
        case_label += f"_{symmetry}"

    print("\n" + "#" * 70)
    print(f"# CASE: {case_label}  (motion={OBSTACLE_MOTION[obstacle]}, regime={regime})")
    print("#" * 70)

    analyzer = ConvergenceAnalyzer(obstacle_instance, obstacle, Ly=cfg.Ly)
    dirs = setup_output_dirs(os.path.join(cfg.output_dir, case_label))

    resolutions = cfg.resolutions
    time_steps = cfg.time_steps_for(regime)
    t_final = cfg.final_time_for(regime)

    results = {n: {"L2": [], "H1": []} for n in resolutions}
    pressure_final = {n: float('nan') for n in resolutions}
    profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    y_ref, u_ref = None, None

    print(f">> Time steps: {len(time_steps)} | Resolutions: {resolutions}\n")

    for t in time_steps:
        print(f" Processing t = {t:.2f}s")

        conf_vel_file = paths.reference_velocity_file(obstacle, Re, t, symmetry)
        if not os.path.exists(conf_vel_file):
            print(f"  [Warning] Reference velocity missing: {conf_vel_file}. Skipping step.")
            for n in resolutions:
                results[n]["L2"].append(float('nan'))
                results[n]["H1"].append(float('nan'))
            continue

        u_ex = load_hdf5_solution(conf_vel_file, "velocity")

        for n in resolutions:
            chk_file = paths.velocity_file(method, obstacle, Re, n, t, symmetry)
            if os.path.exists(chk_file):
                u_h = load_hdf5_solution(chk_file, "velocity")
                err_L2, err_H1 = analyzer.compute_fluid_errors(u_ex, u_h, t)
                results[n]["L2"].append(err_L2)
                results[n]["H1"].append(err_H1)

                if abs(t - t_final) < 1e-8:
                    profiles[n] = analyzer.extract_vertical_profile(u_h, t)
            else:
                print(f"  [Warning] Missing penalized velocity file: {chk_file}")
                results[n]["L2"].append(float('nan'))
                results[n]["H1"].append(float('nan'))

        if abs(t - t_final) < 1e-8:
            y_ref, u_ref = analyzer.extract_vertical_profile(u_ex, t)

            conf_p_file = paths.reference_pressure_file(obstacle, Re, t, symmetry)
            if os.path.exists(conf_p_file):
                p_ex = load_hdf5_solution(conf_p_file, "pressure")
                for n in resolutions:
                    chk_p_file = paths.pressure_file(method, obstacle, Re, n, t, symmetry)
                    if os.path.exists(chk_p_file):
                        p_h = load_hdf5_solution(chk_p_file, "pressure")
                        pressure_final[n] = analyzer.compute_pressure_error_L2(p_ex, p_h, t)
                    else:
                        print(f"  [Warning] Missing penalized pressure file: {chk_p_file}")
            else:
                print(f"  [Warning] Reference pressure missing: {conf_p_file}")
        print()

    summary = _compute_convergence_summary(results, pressure_final, resolutions, time_steps)
    _print_summary(case_label, summary)
    _save_summary(case_label, summary, dirs["data"])
    _save_raw_errors(results, time_steps, resolutions, dirs["data"])
    generate_convergence_plots(case_label, summary, dirs["conv_plots"])

    if y_ref is not None:
        generate_profile_plots(case_label, y_ref, u_ref, profiles, resolutions, dirs["profiles"])

    print(f">> Case '{case_label}' results saved under: {dirs['root']}")
    return summary


# =============================================================================
# 7. CONVERGENCE RATE COMPUTATION
# =============================================================================

def _bochner_norm(err_array: np.ndarray, time_steps: np.ndarray, mask: np.ndarray) -> float:
    """
    Discrete L2(0,T; *) norm via the trapezoidal rule. Falls back to the
    (single) available value when only one time sample exists (steady case).
    """
    valid_times = time_steps[mask]
    valid_err_sq = err_array[mask] ** 2
    if len(valid_times) < 2:
        return math.sqrt(valid_err_sq[0]) if len(valid_err_sq) else float('nan')
    return math.sqrt(np.trapz(valid_err_sq, valid_times))


def _compute_convergence_summary(results, pressure_final, resolutions, time_steps) -> dict:
    dx_h = [1.0 / n for n in resolutions]
    errors_L2_T, errors_H1_T, errors_p_T = [], [], []

    for n in resolutions:
        l2_arr = np.array(results[n]["L2"])
        h1_arr = np.array(results[n]["H1"])
        mask = ~np.isnan(l2_arr)

        if not np.any(mask):
            errors_L2_T.append(float('nan'))
            errors_H1_T.append(float('nan'))
        else:
            errors_L2_T.append(_bochner_norm(l2_arr, time_steps, mask))
            errors_H1_T.append(_bochner_norm(h1_arr, time_steps, mask))

        errors_p_T.append(pressure_final[n])

    rates_L2, rates_H1, rates_p = [], [], []
    for i in range(len(resolutions) - 1):
        log_h = np.log(dx_h[i] / dx_h[i + 1])

        e1_L2, e2_L2 = errors_L2_T[i], errors_L2_T[i + 1]
        e1_H1, e2_H1 = errors_H1_T[i], errors_H1_T[i + 1]
        e1_p, e2_p = errors_p_T[i], errors_p_T[i + 1]

        rates_L2.append(np.log(e1_L2 / e2_L2) / log_h if not any(np.isnan([e1_L2, e2_L2])) else float('nan'))
        rates_H1.append(np.log(e1_H1 / e2_H1) / log_h if not any(np.isnan([e1_H1, e2_H1])) else float('nan'))
        rates_p.append(np.log(e1_p / e2_p) / log_h if not any(np.isnan([e1_p, e2_p])) else float('nan'))

    return {
        "resolutions": resolutions,
        "h": dx_h,
        "L2": errors_L2_T,
        "H1": errors_H1_T,
        "p_L2_final": errors_p_T,
        "rates_L2": rates_L2,
        "rates_H1": rates_H1,
        "rates_p": rates_p,
    }


def _print_summary(case_label: str, s: dict):
    print(f"\n{'=' * 60}\n--- Convergence summary: {case_label} ---\n{'=' * 60}")
    for i, n in enumerate(s["resolutions"]):
        print(f" n={n:4d} | h={s['h'][i]:.4f} | "
              f"L2(0,T;L2)={s['L2'][i]:.5e} | L2(0,T;H1)={s['H1'][i]:.5e} | "
              f"p_L2(t=T)={s['p_L2_final'][i]:.5e}")
    print()
    for i in range(len(s["resolutions"]) - 1):
        n1, n2 = s["resolutions"][i], s["resolutions"][i + 1]
        print(f"  Rate n={n1}->{n2}: "
              f"p_uL2={s['rates_L2'][i]:+.3f}  p_uH1={s['rates_H1'][i]:+.3f}  "
              f"p_pL2={s['rates_p'][i]:+.3f}")


# =============================================================================
# 8. SAVING RESULTS
# =============================================================================

def _save_summary(case_label: str, s: dict, data_dir: str):
    path = os.path.join(data_dir, "convergence_summary.txt")
    with open(path, "w") as f:
        f.write(f"CONVERGENCE ANALYSIS SUMMARY -- {case_label}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'n':>5} {'h':>10} {'L2(0,T;L2)':>15} {'L2(0,T;H1)':>15} {'p_L2(t=T)':>15}\n")
        f.write("-" * 65 + "\n")
        for i, n in enumerate(s["resolutions"]):
            f.write(f"{n:>5} {s['h'][i]:>10.4f} {s['L2'][i]:>15.5e} "
                    f"{s['H1'][i]:>15.5e} {s['p_L2_final'][i]:>15.5e}\n")
        f.write("\nConvergence rates:\n")
        for i in range(len(s["resolutions"]) - 1):
            n1, n2 = s["resolutions"][i], s["resolutions"][i + 1]
            f.write(f"  n={n1}->{n2}: p_L2={s['rates_L2'][i]:+.3f}  "
                    f"p_H1={s['rates_H1'][i]:+.3f}  p_pL2={s['rates_p'][i]:+.3f}\n")
    print(f"   Summary saved: {path}")


def _save_raw_errors(results, time_steps, resolutions, data_dir):
    for n in resolutions:
        path = os.path.join(data_dir, f"errors_n{n}.csv")
        l2 = results[n]["L2"]
        h1 = results[n]["H1"]
        with open(path, "w") as f:
            f.write("t,L2_error,H1_error\n")
            for t, e2, e1 in zip(time_steps, l2, h1):
                f.write(f"{t:.2f},{e2:.6e},{e1:.6e}\n")
    print(f"   Raw error CSVs saved in: {data_dir}")


# =============================================================================
# 9. PLOTTING
# =============================================================================

def generate_convergence_plots(case_label: str, summary: dict, plot_dir: str):
    """Log-log error-vs-h plot (L2 and H1) for a single case."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Spatial Convergence — {case_label}", fontsize=13, fontweight="bold")

    h_vals = np.array(summary["h"])

    for ax, norm_name, tex in zip(axes, ["L2", "H1"], ["L^2", "H^1"]):
        e_vals = np.array(summary[norm_name])
        valid = ~np.isnan(e_vals)
        ax.set_title(f"$L^2(0,T; {tex})$ Error")

        if valid.sum() >= 2:
            ax.loglog(h_vals[valid], e_vals[valid], color="#2c7bb6",
                       marker="o", markersize=7, linewidth=1.8, label=case_label)

            h_ref = np.array([h_vals[valid].max(), h_vals[valid].min()])
            for order, ls, lbl in [(1, "--", "O(h)"), (2, ":", "O(h²)")]:
                scale = e_vals[valid][0] / (h_ref[0] ** order)
                ax.loglog(h_ref, scale * h_ref ** order, color="gray",
                          linestyle=ls, linewidth=1.2, label=lbl)

        ax.set_xlabel("Mesh size $h = 1/n$")
        ax.set_ylabel("Error norm")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(plot_dir, "convergence_loglog.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved: {path}")


def generate_profile_plots(case_label, y_ref, u_ref, profiles, resolutions, plot_dir):
    """Vertical velocity-magnitude profile at the final time, one case."""
    colors = {50: "#f4a61d", 100: "#2ca02c", 150: "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Velocity Magnitude Profile at $t = T$ — {case_label}\n"
                 "Vertical Cut Through Obstacle Center", fontsize=12, fontweight="bold")

    ax.plot(u_ref, y_ref, color="black", linewidth=2.0, linestyle="-",
            label="Conforming (ref.)", zorder=5)

    for n in resolutions:
        if n in profiles:
            y_p, u_p = profiles[n]
            ax.plot(u_p, y_p, color=colors.get(n, "gray"), linewidth=1.5,
                    linestyle="--", label=f"n={n}")

    ax.set_xlabel("$\\|\\mathbf{u}\\|$ (velocity magnitude)")
    ax.set_ylabel("$y$ (vertical coordinate)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(plot_dir, "velocity_profile.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved: {path}")


# =============================================================================
# 10. CASE GENERATION & TOP-LEVEL DRIVER
# =============================================================================

def generate_cases(cfg: AnalysisConfig) -> List[Tuple[str, str, int, Optional[str]]]:
    """
    Expands the configuration into the concrete list of
    (method, obstacle, Reynolds, symmetry) combinations to analyze.
    ``symmetry`` is None whenever that specific (obstacle, Re) combination
    has no symmetry sub-folder (see ``case_has_symmetry``).
    """
    cases = []
    for method in cfg.methods:
        for obstacle in cfg.obstacles:
            for Re in cfg.reynolds:
                if case_has_symmetry(obstacle, Re):
                    for sym in cfg.symmetries:
                        cases.append((method, obstacle, Re, sym))
                else:
                    cases.append((method, obstacle, Re, None))
    return cases


def run_pipeline(cfg: AnalysisConfig, obstacle_instances: Dict[str, object]):
    """
    Runs the full convergence-analysis campaign: every (method, obstacle,
    Reynolds, symmetry) combination described by ``cfg`` is processed and
    written to its own sub-folder of ``cfg.output_dir``.

    obstacle_instances: mapping "cylinder" -> circleObstacle(...) instance,
                         "square"   -> squareObstacle(...) instance.
    """
    paths = PathBuilder(cfg.base_dir, cfg)
    cases = generate_cases(cfg)

    print(f">> Total cases to process: {len(cases)}")

    all_summaries = {}
    for method, obstacle, Re, symmetry in cases:
        obstacle_instance = obstacle_instances[obstacle]
        label = f"{method}_{obstacle}_Re{Re}" + (f"_{symmetry}" if symmetry else "")
        try:
            all_summaries[label] = run_case(
                cfg, method, obstacle, Re, symmetry, obstacle_instance, paths
            )
        except Exception as exc:
            print(f"  [Error] Case '{label}' failed: {exc}")

    print(f"\n>> All cases processed. Results saved under: {cfg.output_dir}/")
    return all_summaries


# =============================================================================
# 11. EXECUTION TARGET
# =============================================================================

if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from domain_settings.obstacles import circleObstacle, squareObstacle

    # Physical obstacle instances (one per obstacle type).
    obstacle_instances = {
        "cylinder": circleObstacle(x=0.5, y=0.5, r=0.1),
        "square":   squareObstacle(x=0.5, y=0.5, side_length=0.2),
    }

    # --- Select here which analysis to run ---------------------------------
    config = AnalysisConfig(
        base_dir=parent_dir,
        output_dir=os.path.join(current_dir, "results"),
        methods=["Brinkman", "DLM"],       # subset allowed, e.g. ["Brinkman"]
        obstacles=["cylinder", "square"],  # subset allowed, e.g. ["square"]
        reynolds=[40],                     # 80 when ready               
        symmetries=["symmetric"],          # asymmetric when ready
        resolutions=[50, 100, 150],
        R_penalty=1000.0,
    )

    run_pipeline(config, obstacle_instances)