import os
from firedrake import *
from firedrake import VTKFile
from firedrake.pyplot import triplot, tripcolor
import matplotlib.pyplot as plt


def save_VTK(file_dict, t, uh, ph, phi, chi):
    """
    Salva i risultati (velocità, pressione, phi, chi) in formato VTK.
    """
    uh.rename('u', 'u')
    ph.rename('p', 'p')
    phi.rename('phi', 'phi')
    chi.rename('chi', 'chi')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    file_dict['phi'].write(phi, time=t)
    file_dict['chi'].write(chi, time=t)


def save_checkpoint(basedir, t_val, uh, ph, phiFun, chiFun, mesh=None, moving=False):
    """
    Salva i risultati e la mesh in file checkpoint (.h5) per il post-processing.
    """
    # Creazione delle sottocartelle se non esistono
    basedir_vel = os.path.join(basedir, 'velocity')
    basedir_pres = os.path.join(basedir, 'pressure')
    basedir_phi = os.path.join(basedir, 'phi')
    basedir_chi = os.path.join(basedir, 'chi')
    basedir_mesh = os.path.join(basedir, 'mesh')

    subdirs = [basedir_vel, basedir_pres, basedir_phi, basedir_chi, basedir_mesh]
    for folder in subdirs:
        os.makedirs(folder, exist_ok=True)

    # Salvataggio delle funzioni
    with CheckpointFile(os.path.join(basedir_vel, f'velocity_t={t_val:.2f}.h5'), 'w') as chk:
        chk.save_function(uh, name='velocity')

    with CheckpointFile(os.path.join(basedir_pres, f'pressure_t={t_val:.2f}.h5'), 'w') as chk:
        chk.save_function(ph, name='pressure')

    with CheckpointFile(os.path.join(basedir_phi, f'phi_t={t_val:.2f}.h5'), 'w') as chk:
        chk.save_function(phiFun, name='phi')

    with CheckpointFile(os.path.join(basedir_chi, f'chi_t={t_val:.2f}.h5'), 'w') as chk:
        chk.save_function(chiFun, name='chi')

    # Salvataggio della mesh
    if mesh:
        mesh_filename = f'mesh_t={t_val:.2f}.h5' if moving else 'mesh.h5'
        with CheckpointFile(os.path.join(basedir_mesh, mesh_filename), 'w') as chk:
            chk.save_mesh(mesh)


def plot_results(mesh, uh, ph, t_val, basedir):
    """
    Crea e salva un'immagine con i plot di mesh, pressione e velocità.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    time_str = f" a t = {t_val:.2f}" if t_val is not None else ""

    # Plot della Mesh
    axes[0].set_title(f"Mesh{time_str}")
    triplot(mesh, axes=axes[0], interior_kw={"color": "k", "linewidth": 0.5})
    axes[0].set_aspect('equal')

    # Plot della Pressione
    axes[1].set_title(f"Pressure (p){time_str}")
    plot_p = tripcolor(ph, axes=axes[1], cmap='coolwarm')
    fig.colorbar(plot_p, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    axes[1].set_aspect('equal')

    # Plot della Velocità
    axes[2].set_title(f"Velocity (u){time_str}")
    V_scalar = FunctionSpace(mesh, "CG", 1)
    u_mag = Function(V_scalar).interpolate(sqrt(inner(uh, uh)))
    plot_u = tripcolor(u_mag, axes=axes[2], cmap='viridis')
    fig.colorbar(plot_u, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    # Aggiunta quiver plot per la direzione della velocità
    coarse_mesh = RectangleMesh(24, 8, mesh.geometric_dimension(), 1)
    V_coarse = VectorFunctionSpace(coarse_mesh, "CG", 1)
    uh_coarse = Function(V_coarse).interpolate(uh, allow_missing_dofs=True)
    x_coarse = coarse_mesh.coordinates.dat.data_ro[:, 0]
    y_coarse = coarse_mesh.coordinates.dat.data_ro[:, 1]
    U_coarse = uh_coarse.dat.data_ro[:, 0]
    V_coarse = uh_coarse.dat.data_ro[:, 1]
    axes[2].quiver(x_coarse, y_coarse, U_coarse, V_coarse,
                   color='black', scale=40, width=0.001, headwidth=2, pivot='mid')
    axes[2].set_aspect('equal')

    plt.tight_layout()

    # Salvataggio della figura
    plot_dir = os.path.join(basedir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'plot_t={t_val:.2f}.png'), dpi=200)
    plt.close(fig)


def create_output_folders(solver_name, params):
    """
    Crea la directory di output e restituisce il percorso base e il dizionario per i file VTK.
    """
    path_parts = ['cyl', solver_name]

    if params.get('moving'):
        path_parts.append('moving')
    else:
        path_parts.append('fixed')

    if params.get('unsteady'):
        path_parts.append('unsteady')
        if params.get('symmetric'):
            path_parts.append('symmetric')
        else:
            path_parts.append('asymmetric')
    else:
        path_parts.append('steady')

    param_string = f"n{params.get('n', 'N')}"
    if 'R' in params:
        param_string += f"_R{params.get('R')}"

    basedir = os.path.join(*path_parts, param_string)
    os.makedirs(basedir, exist_ok=True)

    # Creazione file VTK
    file_dict = {
        'u': VTKFile(os.path.join(basedir, 'velocity.pvd')),
        'p': VTKFile(os.path.join(basedir, 'pressure.pvd')),
        'phi': VTKFile(os.path.join(basedir, 'phi.pvd')),
        'chi': VTKFile(os.path.join(basedir, 'chi.pvd'))
    }
    return basedir, file_dict