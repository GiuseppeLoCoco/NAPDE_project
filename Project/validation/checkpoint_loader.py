"""
Utilities for path resolution, saving, and loading simulation checkpoints in Plots/ directory.
Unifies access across validation and benchmark scripts (convergence_analysis.py, test_L2_penalization_paper.py).
"""

import os
import sys
from typing import Tuple, Optional
import numpy as np
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


from firedrake import CheckpointFile, FunctionSpace, Function


def _safe_load_mesh(chk):
    """Safely loads a mesh from CheckpointFile by inspecting available mesh keys if default fails."""
    try:
        return chk.load_mesh()
    except Exception:
        if hasattr(chk, "h5pyfile") and "meshes" in chk.h5pyfile:
            mesh_names = list(chk.h5pyfile["meshes"].keys())
            if mesh_names:
                return chk.load_mesh(name=mesh_names[0])
        for name in ["mesh", "fluid_mesh", "conforming_mesh", "domain"]:
            try:
                return chk.load_mesh(name=name)
            except Exception:
                pass
        raise


def _safe_load_solution_pair(mesh_file: str, vel_file: str, press_file: str):
    """Loads mesh, velocity, and pressure functions resolving any topology ID discrepancies."""
    # First load mesh and velocity from vel_file
    with CheckpointFile(vel_file, 'r') as chk_v:
        mesh = _safe_load_mesh(chk_v)
        u = chk_v.load_function(mesh, name="velocity")

    # Load pressure function
    with CheckpointFile(press_file, 'r') as chk_p:
        try:
            p = chk_p.load_function(mesh, name="pressure")
        except Exception:
            mesh_p = _safe_load_mesh(chk_p)
            p_raw = chk_p.load_function(mesh_p, name="pressure")
            V_p = FunctionSpace(mesh, p_raw.function_space().ufl_element())
            p = Function(V_p, name="pressure")
            p.dat.data[:] = p_raw.dat.data_ro[:]

    return mesh, u, p


def load_conforming_solution(obstacle_type: str = "square", n: int = 320, Re: float = 40.0, t_final: Optional[float] = None, **kwargs):
    """
    Checks for an existing conforming reference simulation in Plots/Conforming/fixed/<obstacle>/symmetric/n{n}_Re{Re}/
    matching n and Re, taking the checkpoint at t = t_final (if specified) or latest available time step t.
    Returns (ref_mesh, u_ref, p_ref) or (None, None, None).
    """
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    conforming_base = os.path.join(project_dir, "Plots", "Conforming", "fixed", obstacle_type, sym_str)

    if os.path.exists(conforming_base):
        for folder in os.listdir(conforming_base):
            if folder.startswith(f"n{n}_") and (f"Re{Re}" in folder or f"Re{int(Re)}" in folder or f"Re{Re:.1f}" in folder):
                base_dir = os.path.join(conforming_base, folder)
                mesh_file = os.path.join(base_dir, "mesh", "mesh.h5")

                if t_final is not None:
                    vel_file = os.path.join(base_dir, "velocity", f"velocity_t={t_final:.2f}.h5")
                    press_file = os.path.join(base_dir, "pressure", f"pressure_t={t_final:.2f}.h5")
                    if os.path.exists(vel_file) and os.path.exists(press_file):
                        try:
                            mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                            print(f"\n--- Found and loaded Conforming reference (t={t_final:.2f}s) from {base_dir} ---")
                            return mesh, u, p
                        except Exception as e:
                            print(f"Warning: Error loading Conforming from {base_dir}: {e}")
                else:
                    mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                    if vel_file is not None and press_file is not None:
                        try:
                            mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                            print(f"\n--- Found and loaded Conforming reference (t={t_found:.2f}s) from {base_dir} ---")
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
                        if os.path.exists(vel_file) and os.path.exists(press_file):
                            try:
                                mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                                print(f"--- Found and loaded Brinkman solution (R={R_val:.1e}, t={t_final:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception as e:
                                print(f"Warning: Error loading Brinkman from {base_dir}: {e}")
                    else:
                        mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                        if vel_file is not None and press_file is not None:
                            try:
                                mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                                print(f"--- Found and loaded Brinkman solution (R={R_val:.1e}, t={t_found:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception:
                                pass
    return None, None, None


def load_dlm_solution(obstacle_type: str = "square", n: int = 320, Re: float = 40.0, t_final: Optional[float] = None, **kwargs):
    """
    Checks for an existing DLM simulation in Plots/DLM/fixed/<obstacle>/symmetric/n{n}_Re{Re}/.
    If t_final is specified, requires an exact match at t = t_final (returns None if not available).
    Returns (mesh, uh, ph) or (None, None, None).
    """
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    dlm_base = os.path.join(project_dir, "Plots", "DLM", "fixed", obstacle_type, sym_str)

    if os.path.exists(dlm_base):
        for folder in os.listdir(dlm_base):
            if folder.startswith(f"n{n}_") and (f"Re{Re}" in folder or f"Re{int(Re)}" in folder or f"Re{Re:.1f}" in folder):
                base_dir = os.path.join(dlm_base, folder)
                mesh_file = os.path.join(base_dir, "mesh", "mesh.h5")

                if t_final is not None:
                    vel_file = os.path.join(base_dir, "velocity", f"velocity_t={t_final:.2f}.h5")
                    press_file = os.path.join(base_dir, "pressure", f"pressure_t={t_final:.2f}.h5")
                    if os.path.exists(vel_file):
                        try:
                            mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file if os.path.exists(press_file) else None)
                            print(f"--- Found and loaded DLM solution (n={n}, Re={Re}, t={t_final:.2f}s) from {base_dir} ---")
                            return mesh, u, p
                        except Exception as e:
                            print(f"Warning: Error loading DLM from {base_dir}: {e}")
                else:
                    mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                    if vel_file is not None:
                        try:
                            mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                            print(f"--- Found and loaded DLM solution (n={n}, Re={Re}, t={t_found:.2f}s) from {base_dir} ---")
                            return mesh, u, p
                        except Exception:
                            pass
    return None, None, None


def load_riis_solution(obstacle_type: str = "square", n: int = 320, R_val: float = 1000.0, Re: float = 40.0, t_final: Optional[float] = None, **kwargs):
    """
    Checks for an existing RIIS simulation in Plots/RIIS/fixed/<obstacle>/symmetric/n{n}_R{R}_Re{Re}/.
    If t_final is specified, requires an exact match at t = t_final (returns None if not available).
    Returns (mesh, uh, ph) or (None, None, None).
    """
    sym_str = "symmetric" if abs(y_obs - 0.5 * Ly) < 1e-6 else "asymmetric"
    riis_base = os.path.join(project_dir, "Plots", "RIIS", "fixed", obstacle_type, sym_str)

    if os.path.exists(riis_base):
        for folder in os.listdir(riis_base):
            if folder.startswith(f"n{n}_") and (f"Re{Re}" in folder or f"Re{int(Re)}" in folder or f"Re{Re:.1f}" in folder):
                r_matches = [f"_R{R_val}_", f"_R{int(R_val)}_" if R_val >= 1 and R_val == int(R_val) else f"_R{R_val:.1e}_", f"_R{R_val:.1f}_", f"_R{R_val}"]
                if any(rm in folder for rm in r_matches) or folder.endswith(f"_R{R_val}"):
                    base_dir = os.path.join(riis_base, folder)
                    mesh_file = os.path.join(base_dir, "mesh", "mesh.h5")

                    if t_final is not None:
                        vel_file = os.path.join(base_dir, "velocity", f"velocity_t={t_final:.2f}.h5")
                        press_file = os.path.join(base_dir, "pressure", f"pressure_t={t_final:.2f}.h5")
                        if os.path.exists(vel_file):
                            try:
                                mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file if os.path.exists(press_file) else None)
                                print(f"--- Found and loaded RIIS solution (R={R_val:.1e}, t={t_final:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception as e:
                                print(f"Warning: Error loading RIIS from {base_dir}: {e}")
                    else:
                        mesh_file, vel_file, press_file, t_found = find_latest_checkpoint_in_dir(base_dir)
                        if vel_file is not None:
                            try:
                                mesh, u, p = _safe_load_solution_pair(mesh_file, vel_file, press_file)
                                print(f"--- Found and loaded RIIS solution (R={R_val:.1e}, t={t_found:.2f}s) from {base_dir} ---")
                                return mesh, u, p
                            except Exception:
                                pass
    return None, None, None




def extract_probe_history(case_dir: str, probe_pt: Tuple[float, float] = (1.5, 0.5)):
    """
    Extracts time history of velocity (u_x, u_y) at probe_pt across all saved velocity checkpoints.
    Returns (times, u_x_series, u_y_series) sorted by time.
    """
    vel_dir = os.path.join(case_dir, "velocity")
    if not os.path.isdir(vel_dir):
        return np.array([]), np.array([]), np.array([])

    files = [f for f in os.listdir(vel_dir) if f.startswith("velocity_t=") and f.endswith(".h5")]
    if not files:
        return np.array([]), np.array([]), np.array([])

    timed_files = []
    for f in files:
        t_str = f[len("velocity_t="):-len(".h5")]
        try:
            timed_files.append((float(t_str), os.path.join(vel_dir, f)))
        except ValueError:
            pass

    timed_files.sort(key=lambda x: x[0])
    times = []
    ux_list = []
    uy_list = []

    import warnings
    mesh_cached = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for t_val, fpath in timed_files:
            try:
                with CheckpointFile(fpath, 'r') as chk:
                    if mesh_cached is None:
                        mesh_cached = _safe_load_mesh(chk)
                    u_func = chk.load_function(mesh_cached, name="velocity")
                    val = u_func.at(probe_pt)
                    times.append(t_val)
                    ux_list.append(float(val[0]))
                    uy_list.append(float(val[1]))
            except Exception:
                continue

    return np.array(times), np.array(ux_list), np.array(uy_list)
