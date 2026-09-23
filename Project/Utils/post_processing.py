import os
from firedrake import *
from firedrake.pyplot import triplot, tripcolor
import matplotlib.pyplot as plt


import glob
import re
import xml.etree.ElementTree as ET


# Global storage for PVD history when resuming simulations
_PVD_HISTORY = {}


def _safe_load_mesh(chk):
    """Safely loads a mesh from CheckpointFile by inspecting available mesh keys."""
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


def find_latest_checkpoint(basedir: str):
    """
    Finds the latest available and uncorrupted checkpoint (maximum t) in basedir.
    Returns (latest_t, vel_file, press_file, mesh_file) or (None, None, None, None).
    """
    if not os.path.isdir(basedir):
        return None, None, None, None

    vel_dir = os.path.join(basedir, "velocity")
    if not os.path.isdir(vel_dir):
        return None, None, None, None

    press_dir = os.path.join(basedir, "pressure")
    mesh_dir = os.path.join(basedir, "mesh")

    # Find candidate velocity files
    vel_files = [f for f in os.listdir(vel_dir) if f.startswith("velocity_t=") and f.endswith(".h5")]
    if not vel_files:
        return None, None, None, None

    candidates = []
    for vf in vel_files:
        m = re.search(r"velocity_t=([0-9.]+)\.h5", vf)
        if m:
            try:
                t_val = float(m.group(1))
                candidates.append((t_val, os.path.join(vel_dir, vf)))
            except ValueError:
                continue

    if not candidates:
        return None, None, None, None

    # Sort descending by time
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Check each candidate from latest to earliest until a valid uncorrupted checkpoint is found
    for t_val, vel_path in candidates:
        # Check pressure file
        press_path = os.path.join(press_dir, f"pressure_t={t_val:.2f}.h5")
        if not os.path.exists(press_path):
            press_path = None

        # Check mesh file
        mesh_path = os.path.join(mesh_dir, f"mesh_t={t_val:.2f}.h5")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(mesh_dir, "mesh.h5")
            if not os.path.exists(mesh_path):
                mesh_path = vel_path  # Mesh is embedded inside velocity checkpoint file

        # Test if velocity checkpoint is uncorrupted and can be opened
        try:
            with CheckpointFile(vel_path, 'r') as chk:
                _ = _safe_load_mesh(chk)
            # If pressure path exists, test it too
            if press_path is not None:
                try:
                    with CheckpointFile(press_path, 'r') as chk_p:
                        _ = _safe_load_mesh(chk_p)
                except Exception:
                    # If pressure file is corrupt, try older checkpoint
                    continue
            return t_val, vel_path, press_path, mesh_path
        except Exception:
            # Corrupted / partially written file, try previous
            continue

    return None, None, None, None


def load_checkpoint_solution(vel_file: str, press_file: str = None, target_u=None, target_p=None):
    """
    Loads velocity and pressure functions from checkpoint files.
    If target_u / target_p are provided, updates their values in-place.
    Returns (mesh, u, p).
    """
    with CheckpointFile(vel_file, 'r') as chk_v:
        mesh = _safe_load_mesh(chk_v)
        u_loaded = chk_v.load_function(mesh, name="velocity")

    p_loaded = None
    if press_file is not None and os.path.exists(press_file):
        with CheckpointFile(press_file, 'r') as chk_p:
            try:
                p_loaded = chk_p.load_function(mesh, name="pressure")
            except Exception:
                mesh_p = _safe_load_mesh(chk_p)
                p_loaded = chk_p.load_function(mesh_p, name="pressure")

    # Assign to target_u if provided
    if target_u is not None:
        try:
            target_u.assign(u_loaded)
        except Exception:
            try:
                target_u.dat.data[:] = u_loaded.dat.data_ro[:]
            except Exception:
                target_u.interpolate(u_loaded, allow_missing_dofs=True)

    # Assign to target_p if provided
    if target_p is not None and p_loaded is not None:
        try:
            target_p.assign(p_loaded)
        except Exception:
            try:
                target_p.dat.data[:] = p_loaded.dat.data_ro[:]
            except Exception:
                target_p.interpolate(p_loaded, allow_missing_dofs=True)

    return mesh, (target_u if target_u is not None else u_loaded), (target_p if target_p is not None else p_loaded)


def _read_pvd_entries_up_to(pvd_path: str, max_t: float):
    """Reads existing <DataSet> entries from a .pvd file up to max_t."""
    entries = []
    if not os.path.exists(pvd_path):
        return entries
    try:
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        collection = root.find("Collection")
        if collection is not None:
            for ds in collection.findall("DataSet"):
                try:
                    ts = float(ds.get("timestep", -1.0))
                    if ts <= max_t + 1e-6:
                        entries.append((ts, ds.get("file", "")))
                except ValueError:
                    pass
    except Exception:
        pass
    return entries


def _merge_pvd_file(pvd_path: str, prior_entries):
    """Ensures prior <DataSet> entries are preserved in the .pvd file after writing."""
    if not prior_entries or not os.path.exists(pvd_path):
        return
    try:
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        collection = root.find("Collection")
        if collection is not None:
            combined = list(prior_entries)
            existing_times = {pt[0] for pt in combined}
            for ds in collection.findall("DataSet"):
                try:
                    ts = float(ds.get("timestep", -9999.0))
                    f_val = ds.get("file", "")
                    if not any(abs(ts - et) < 1e-6 for et in existing_times):
                        combined.append((ts, f_val))
                        existing_times.add(ts)
                except ValueError:
                    pass
            combined.sort(key=lambda x: x[0])
            collection.clear()
            for ts, f_val in combined:
                ds_elem = ET.SubElement(collection, "DataSet")
                ds_elem.set("timestep", str(ts))
                ds_elem.set("file", f_val)
            tree.write(pvd_path, encoding='utf-8', xml_declaration=True)
    except Exception:
        pass


def setup_pvd_resume(basedir: str, file_dict: dict, latest_t: float):
    """Prepares VTK file dict to resume smoothly from latest_t, preserving existing timesteps."""
    if latest_t is None or latest_t < 0:
        return

    for key, vtk_obj in file_dict.items():
        field_name = 'velocity' if key == 'u' else ('pressure' if key == 'p' else key)
        pvd_path = os.path.join(basedir, f"{field_name}.pvd")
        prior_entries = _read_pvd_entries_up_to(pvd_path, latest_t)
        if prior_entries:
            _PVD_HISTORY[pvd_path] = prior_entries

            # Set the internal counter of VTKFile to avoid overwriting earlier .vtu files
            max_idx = -1
            for _, f_str in prior_entries:
                m = re.search(r"_(\d+)\.vtu", f_str)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
            next_count = max_idx + 1 if max_idx >= 0 else len(prior_entries)

            for attr in ['_count', '_counter', 'counter', 'count', '_idx', '_index', '_step']:
                if hasattr(vtk_obj, attr):
                    setattr(vtk_obj, attr, next_count)


def save_VTK(file_dict, t, uh, ph, **kwargs):
    """
    Salva i risultati (velocità, pressione, e campi extra) in formato VTK.
    Preserva la cronologia PVD se la simulazione è stata ripresa.
    """
    uh.rename('u', 'u')
    ph.rename('p', 'p')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    for name, field in kwargs.items():
        field.rename(name, name)
        file_dict[name].write(field, time=t)

    # If this run has prior PVD history from a resume, merge it
    for key, vtk_obj in file_dict.items():
        field_name = 'velocity' if key == 'u' else ('pressure' if key == 'p' else key)
        pvd_path = getattr(vtk_obj, 'filename', None)
        if not pvd_path or not os.path.exists(pvd_path):
            # Try to infer pvd_path from object attributes or history keys
            matched_keys = [k for k in _PVD_HISTORY if k.endswith(f"{field_name}.pvd")]
            if matched_keys:
                pvd_path = matched_keys[0]

        if pvd_path and pvd_path in _PVD_HISTORY:
            _merge_pvd_file(pvd_path, _PVD_HISTORY[pvd_path])


def save_checkpoint(basedir, t_val, mesh=None, moving=False, **kwargs):
    """
    Salva i risultati e la mesh in file checkpoint (.h5) per il post-processing.
    Utilizza scrittura atomica (.tmp -> .h5) per proteggere i dati da interruzioni accidentali.
    """
    # Salvataggio delle funzioni passate come kwargs
    for name, function in kwargs.items():
        if function is None:
            continue
        field_dir = os.path.join(basedir, name)
        os.makedirs(field_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(field_dir, f'{name}_t={t_val:.2f}.h5')
        temp_path = checkpoint_path + ".tmp"
        try:
            with CheckpointFile(temp_path, 'w') as chk:
                chk.save_function(function, name=name)
            os.replace(temp_path, checkpoint_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    # Salvataggio della mesh
    if mesh:
        basedir_mesh = os.path.join(basedir, 'mesh')
        os.makedirs(basedir_mesh, exist_ok=True)
        mesh_filename = f'mesh_t={t_val:.2f}.h5' if moving else 'mesh.h5'
        mesh_path = os.path.join(basedir_mesh, mesh_filename)

        # Se la mesh è fissa e il file esiste già, non è necessario riscriverlo ad ogni passo
        if not (not moving and os.path.exists(mesh_path)):
            temp_mesh = mesh_path + ".tmp"
            try:
                with CheckpointFile(temp_mesh, 'w') as chk:
                    chk.save_mesh(mesh)
                os.replace(temp_mesh, mesh_path)
            except Exception:
                if os.path.exists(temp_mesh):
                    try:
                        os.remove(temp_mesh)
                    except OSError:
                        pass
                raise


def plot_results(mesh, uh, ph, t_val, basedir, solid_mesh=None):
    """
    Crea e salva un'immagine con i plot di mesh, pressione e velocità.
    Se passato `solid_mesh`, sovrappone la geometria del cilindro solido in rosso.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    time_str = f" a t = {t_val:.2f}" if t_val is not None else ""

    x_coords = mesh.coordinates.dat.data_ro[:, 0]
    y_coords = mesh.coordinates.dat.data_ro[:, 1]
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # Plot della Mesh Fluida
    axes[0].set_title(f"Mesh{time_str}")
    triplot(mesh, axes=axes[0], interior_kw={"color": "lightgray", "linewidth": 0.05})
    # triplot(mesh, axes=axes[0], interior_kw={"color": "k", "linewidth": 0.1, "alpha": 0.2})
    
    # Plot della Pressione
    axes[1].set_title(f"Pressure (p){time_str}")
    plot_p = tripcolor(ph, axes=axes[1], cmap='coolwarm')
    fig.colorbar(plot_p, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Plot della Velocità
    axes[2].set_title(f"Velocity (u){time_str}")
    V_scalar = FunctionSpace(mesh, "CG", 1)
    u_mag = Function(V_scalar).interpolate(sqrt(inner(uh, uh)))
    plot_u = tripcolor(u_mag, axes=axes[2], cmap='viridis')
    fig.colorbar(plot_u, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    # Sovrapposizione del cilindro solido (se presente)

    if solid_mesh is not None:
        sm = solid_mesh.mesh if hasattr(solid_mesh, 'mesh') else solid_mesh
        for ax in axes:
            triplot(sm, axes=ax, interior_kw={"color": "red", "linewidth": 0.6})


    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Salvataggio della figura
    plot_dir = os.path.join(basedir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'plot_t={t_val:.2f}.png'), dpi=200)
    plt.close(fig)


def create_output_folders(solver_name, params, extra_fields=None):
    """
    Crea la directory di output e restituisce il percorso base e il dizionario per i file VTK.
    """
    extra_fields = extra_fields or []
    
    # Costruisce un percorso robusto partendo dalla directory del progetto (Project/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if params.get('is_mms'):
        base_plot_dir = os.path.join(project_root, 'Plots', 'MMS', solver_name)
    else:
        base_plot_dir = os.path.join(project_root, 'Plots', solver_name)
    path_parts = [base_plot_dir]

    if params.get('moving'):
        path_parts.append('moving')
    else:
        path_parts.append('fixed')

    obstacle_name = str(params.get('obstacle')).lower() if params.get('obstacle') else None
    if obstacle_name and obstacle_name not in ["none"]:
        path_parts.append(str(params.get('obstacle')))

    # Add symmetric/asymmetric only for bluff body obstacles (cylinder, square) where y_obs position matters
    if obstacle_name in ["cylinder", "square", "circle"]:
        if params.get('symmetric') is False:
            path_parts.append('asymmetric')
        else:
            path_parts.append('symmetric')

    param_string = f"n{params.get('n', 'N')}"
    if 'R' in params:
        param_string += f"_R{params.get('R')}"

    if 'Re' in params:
        param_string += f"_Re{params.get('Re')}"

    basedir = os.path.join(base_plot_dir, *path_parts[1:], param_string)
    os.makedirs(basedir, exist_ok=True)

    # Creazione file VTK
    file_dict = {
        'u': VTKFile(os.path.join(basedir, 'velocity.pvd')),
        'p': VTKFile(os.path.join(basedir, 'pressure.pvd'))
    }
    for field in extra_fields:
        file_dict[field] = VTKFile(os.path.join(basedir, f'{field}.pvd'))

    return basedir, file_dict