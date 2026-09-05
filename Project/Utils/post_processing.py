import os
from firedrake import *
from firedrake.pyplot import triplot, tripcolor
import matplotlib.pyplot as plt


def save_VTK(file_dict, t, uh, ph, **kwargs):
    """
    Salva i risultati (velocità, pressione, e campi extra) in formato VTK.
    """
    uh.rename('u', 'u')
    ph.rename('p', 'p')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    for name, field in kwargs.items():
        field.rename(name, name)
        file_dict[name].write(field, time=t)


def save_checkpoint(basedir, t_val, mesh=None, moving=False, **kwargs):
    """
    Salva i risultati e la mesh in file checkpoint (.h5) per il post-processing.
    I campi da salvare vengono passati come keyword arguments (es. uh=velocity_function).
    """
    # Salvataggio delle funzioni passate come kwargs
    for name, function in kwargs.items():
        # Crea la sottocartella per il campo specifico
        field_dir = os.path.join(basedir, name)
        os.makedirs(field_dir, exist_ok=True)
        
        # Salva la funzione
        checkpoint_path = os.path.join(field_dir, f'{name}_t={t_val:.2f}.h5')
        with CheckpointFile(checkpoint_path, 'w') as chk:
            chk.save_function(function, name=name)

    # Salvataggio della mesh
    if mesh:
        basedir_mesh = os.path.join(basedir, 'mesh')
        os.makedirs(basedir_mesh, exist_ok=True)
        mesh_filename = f'mesh_t={t_val:.2f}.h5' if moving else 'mesh.h5'
        with CheckpointFile(os.path.join(basedir_mesh, mesh_filename), 'w') as chk:
            chk.save_mesh(mesh)


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
    
    obs_raw = params.get('obstacle')
    obs_str = str(obs_raw).lower() if obs_raw is not None else "none"
    is_buffer = obs_str in ["buffer", "none"] and (obs_str == "buffer" or params.get('is_mms'))

    mesh_type = "structured" if params.get('structured', True) else "unstructured"

    param_string = f"n{params.get('n', 'N')}"
    if 'R' in params:
        param_string += f"_R{params.get('R')}"
    if 'Re' in params:
        param_string += f"_Re{params.get('Re')}"

    if is_buffer:
        # Percorso per esperimenti Buffer: Plots/Buffer/{solver_name}/{structured|unstructured}/n{n}_...
        basedir = os.path.join(project_root, 'Plots', 'Buffer', solver_name, mesh_type, param_string)
    else:
        # Percorso standard (ostacoli): Plots/{solver_name}/{structured|unstructured}/fixed/{obstacle}/{sym_str}/n{n}_...
        if params.get('is_mms'):
            base_plot_dir = os.path.join(project_root, 'Plots', 'MMS', solver_name, mesh_type)
        else:
            base_plot_dir = os.path.join(project_root, 'Plots', solver_name, mesh_type)

        path_parts = [base_plot_dir]

        if params.get('moving'):
            path_parts.append('moving')
        else:
            path_parts.append('fixed')

        if params.get('obstacle'):
            path_parts.append(str(params.get('obstacle')))

        if params.get('symmetric') is False:
            path_parts.append('asymmetric')
        else:
            path_parts.append('symmetric')

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