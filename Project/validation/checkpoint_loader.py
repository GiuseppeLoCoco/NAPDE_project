"""
Utilities for path resolution, saving, and loading simulation checkpoints in Plots/ directory.
Unifies access across validation and benchmark scripts (convergence_analysis.py, test_L2_penalization_paper.py).
"""

import os
import sys
from typing import Tuple, Optional
from firedrake import CheckpointFile

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from user_inputs.user_parameters import y_obs, Ly


# =============================================================================
# 1. PATH RESOLUTION FUNCTIONS
# =============================================================================

def get_case_directory(solver_name: str, obstacle: str, Re: float, n: int, R_penalty: float = 1000.0) -> str:
    """Returns the output directory path where solver checkpoints are saved."""
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    param_str = f"n{n}_R{R_penalty}_Re{Re}" if solver_name.lower() == "brinkman" else f"n{n}_Re{Re}"
    return os.path.join(project_dir, "Plots", solver_name, "fixed", obstacle, sym_str, param_str)


def get_field_filepath(case_dir: str, field_name: str, t_val: float) -> str:
    """Finds the .h5 checkpoint filepath for a given field at time t_val."""
    return os.path.join(case_dir, field_name, f"{field_name}_t={t_val:.2f}.h5")


def load_hdf5_solution(filepath: str, field_name: str):
    """Loads Firedrake function from CheckpointFile .h5 file."""
    if not filepath.endswith(".h5"):
        filepath += ".h5"

    with CheckpointFile(filepath, 'r') as chk:
        mesh = chk.load_mesh()
        field = chk.load_function(mesh, field_name)
    return field


# =============================================================================
# 2. CHECKPOINT SEARCH & AUTO-DISCOVERY
# =============================================================================

def find_latest_checkpoint_in_dir(base_dir: str):
    """
    Finds the latest available checkpoint (maximum t) in base_dir containing mesh, velocity, and pressure.
    Returns (mesh_file, velocity_file, pressure_file, latest_t) or (None, None, None, None).
    """
    mesh_file = os.path.join(base_dir, "mesh", "mesh.h5")
    vel_dir = os.path.join(base_dir, "velocity")
    press_dir = os.path.join(base_dir, "pressure")

    if not (os.path.exists(mesh_file) and os.path.isdir(vel_dir) and os.path.isdir(press_dir)):
        return None, None, None, None

    vel_files = [f for f in os.listdir(vel_dir) if f.startswith("velocity_t=") and f.endswith(".h5")]
    if not vel_files:
        return None, None, None, None

    candidates = []
    for vf in vel_files:
        t_str = vf[len("velocity_t="):-len(".h5")]
        pf = f"pressure_t={t_str}.h5"
        press_path = os.path.join(press_dir, pf)
        if os.path.exists(press_path):
            try:
                t_val = float(t_str)
                candidates.append((t_val, os.path.join(vel_dir, vf), press_path))
            except ValueError:
                continue

    if not candidates:
        return None, None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_t, best_vel, best_press = candidates[0]
    return mesh_file, best_vel, best_press, latest_t


def load_conforming_solution(obstacle_type: str = "square", n: int = 320, Re: float = 40.0, **kwargs):
    """
    Checks for an existing conforming reference simulation in Plots/Conforming/fixed/<obstacle>/symmetric/n{n}_Re{Re}/
    matching n and Re, taking the latest available time step t.
    Returns (ref_mesh, u_ref, p_ref) or (None, None, None).
    """
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    conforming_base = os.path.join(project_dir, "Plots", "Conforming", "fixed", obstacle_type, sym_str)

    if os.path.exists(conforming_base):
        for folder in os.listdir(conforming_base):
            if folder.startswith(f"n{n}_") and (f"Re{Re}" in folder or f"Re{int(Re)}" in folder or f"Re{Re:.1f}" in folder):
                base_dir = os.path.join(conforming_base, folder)
                mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                if mesh_file is not None:
                    try:
                        with CheckpointFile(mesh_file, 'r') as chk:
                            mesh = chk.load_mesh()
                        with CheckpointFile(vel_file, 'r') as chk:
                            u = chk.load_function(mesh, name="velocity")
                        with CheckpointFile(press_file, 'r') as chk:
                            p = chk.load_function(mesh, name="pressure")
                        print(f"\n--- Found and loaded Conforming reference (t={t_found:.2f}) from {base_dir} ---")
                        return mesh, u, p
                    except Exception as e:
                        print(f"Warning: Error loading from {base_dir}: {e}")

    return None, None, None


def load_brinkman_solution(obstacle_type: str = "square", n: int = 320, R_val: float = 1000.0, Re: float = 40.0, t_final: Optional[float] = None, **kwargs):
    """
    Checks for an existing Brinkman simulation in Plots/Brinkman/fixed/<obstacle>/symmetric/n{n}_R{R}_Re{Re}/.
    If t_final is specified, requires an exact match at t = t_final (returns None if not available).
    Returns (mesh, uh, ph) or (None, None, None).
    """
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    brinkman_base = os.path.join(project_dir, "Plots", "Brinkman", "fixed", obstacle_type, sym_str)

    if os.path.exists(brinkman_base):
        for folder in os.listdir(brinkman_base):
            if folder.startswith(f"n{n}_") and (f"Re{Re}" in folder or f"Re{int(Re)}" in folder or f"Re{Re:.1f}" in folder):
                r_matches = [f"_R{R_val}_", f"_R{int(R_val)}_" if R_val >= 1 and R_val == int(R_val) else f"_R{R_val:.1e}_", f"_R{R_val:.1f}_"]
                if any(rm in folder for rm in r_matches) or f"_R{R_val}" in folder:
                    base_dir = os.path.join(brinkman_base, folder)
                    mesh_file = os.path.join(base_dir, "mesh", "mesh.h5")

                    if t_final is not None:
                        vel_file = os.path.join(base_dir, "velocity", f"velocity_t={t_final:.2f}.h5")
                        press_file = os.path.join(base_dir, "pressure", f"pressure_t={t_final:.2f}.h5")
                        if os.path.exists(mesh_file) and os.path.exists(vel_file) and os.path.exists(press_file):
                            try:
                                with CheckpointFile(mesh_file, 'r') as chk:
                                    mesh = chk.load_mesh()
                                with CheckpointFile(vel_file, 'r') as chk:
                                    u = chk.load_function(mesh, name="velocity")
                                with CheckpointFile(press_file, 'r') as chk:
                                    p = chk.load_function(mesh, name="pressure")
                                print(f"--- Found and loaded Brinkman solution (R={R_val:.1e}, t={t_final:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception as e:
                                print(f"Warning: Error loading Brinkman from {base_dir}: {e}")
                    else:
                        mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                        if mesh_file is not None:
                            try:
                                with CheckpointFile(mesh_file, 'r') as chk:
                                    mesh = chk.load_mesh()
                                with CheckpointFile(vel_file, 'r') as chk:
                                    u = chk.load_function(mesh, name="velocity")
                                with CheckpointFile(press_file, 'r') as chk:
                                    p = chk.load_function(mesh, name="pressure")
                                print(f"--- Found and loaded Brinkman solution (R={R_val:.1e}, t={t_found:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception:
                                pass
    return None, None, None
