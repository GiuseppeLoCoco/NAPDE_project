import os
import math
import numpy as np
import matplotlib.pyplot as plt
from firedrake import (Mesh, FunctionSpace, Function, VectorFunctionSpace, PointEvaluator,
    CheckpointFile, Constant, project, assemble, dx, inner, grad, sqrt, FILE_READ)
import warnings

# =============================================================================
# 1. POST-PROCESSING & ANALYSIS CLASSES
# =============================================================================

class ConvergenceAnalyzer:

    def __init__(self, obstacle_instance, Ly=1.0):
        """
        Handles convergence analysis and profile extraction for fluid subdomains.
        """
        self.obstacle = obstacle_instance
        self.Ly = Ly

    def compute_fluid_errors(self, u_ex, u_h, t_val):
        """
        Computes L2 and H1 error norms strictly within the fluid domain
        by masking the solid domain.
        """
        V_ex = u_ex.function_space()
        mesh_ex = V_ex.mesh()
        t_const = Constant(t_val)

        # Project approximate solution onto the exact (conforming) space
        # Note: In HPC applications, pre-allocating u_h_proj is preferred.
        u_h_proj = project(u_h, V_ex)

        # Fluid mask (1 in fluid, 0 in solid)
        chi_solid = self.obstacle.chi(mesh_ex, t_const)
        mask_fluid = 1.0 - chi_solid

        # Error field
        err = u_h_proj - u_ex

        # Assemble masked errors
        err_L2_sq = assemble(mask_fluid * inner(err, err) * dx)
        err_H1_sq = assemble(mask_fluid * (inner(err, err) + inner(grad(err), grad(err))) * dx)

        return sqrt(err_L2_sq), sqrt(err_H1_sq)

    def extract_vertical_profile(self, u_h, t_val, num_points=200):
        """
        Extracts velocity magnitude profile along a vertical line passing through
        the cylinder's center. Accounts for rigid-body velocity inside conforming holes.
        """
        # Calculate cylinder center X-coordinate via pure Python math
        displ_x = self.obstacle.amplitude * 0.5 * (1.0 - math.cos(0.2 * math.pi * t_val))
        x_c = self.obstacle.x_obs + displ_x

        # Calculate prescribed solid velocity component (dx/dt)
        # x(t) = x_0 + A/2 * (1 - cos(omega*t)) -> v(t) = A/2 * omega * sin(omega*t)
        v_solid_x = self.obstacle.amplitude * 0.5 * (0.2 * math.pi) * math.sin(0.2 * math.pi * t_val)

        y_coords = np.linspace(1e-5, self.Ly - 1e-5, num_points)
        u_mag = np.zeros_like(y_coords)

        for i, y in enumerate(y_coords):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    val = u_h.at([x_c, y], tolerance=1e-6)
                u_mag[i] = math.sqrt(val[0]**2 + val[1]**2)
            except Exception:
                # Punto dentro il foro della mesh conforming → velocità rigida
                u_mag[i] = abs(v_solid_x)

        return y_coords, u_mag


# OUTPUT DIRECTORY SETUP

def setup_output_dirs(base_output_dir):
    """
    Crea la struttura di cartelle per i risultati:
      results/
        plots/
          profiles/
          convergence/
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

# =============================================================================
# 2. DATA LOADING & PIPELINE AUTOMATION
# =============================================================================

def load_hdf5_solution(checkpoint_path, field_name="velocity"):
    """
    Loads a saved Firedrake function from a DumbCheckpoint .h5 file.
    """
    # Stripping the .h5 extension if Firedrake's DumbCheckpoint appends it automatically
    if checkpoint_path.endswith(".h5"):
        checkpoint_path = checkpoint_path[:-3]

    if not checkpoint_path.endswith(".h5"):
        checkpoint_path = checkpoint_path + ".h5"

    with CheckpointFile(checkpoint_path, 'r') as chk:
        mesh = chk.load_mesh()          # carica la mesh salvata nel file
        u = chk.load_function(mesh, field_name)   # carica il campo per nome

    return u


def run_pipeline(base_dir, obstacle_model, output_dir="results"):
    """
    Automates directory traversal, error assessment, and plotting.
    """
    analyzer = ConvergenceAnalyzer(obstacle_model)
    dirs = setup_output_dirs(output_dir)

    # Simulation parameters
    t_start, t_end, dt = 0.5, 10.0, 0.5
    time_steps = np.arange(t_start, t_end + dt, dt)
    resolutions = [25, 50, 70]
    methods = ["brinkman", "RIIS"]
    
    # Data storage structure
    results = {method: {n: {"L2": [], "H1": []} for n in resolutions} for method in methods}
    profiles = {method: {} for method in methods}
    y_ref    = None
    u_ref    = None

    print(">> Starting Spatial Convergence Analysis...")
    print(f"   Time steps: {len(time_steps)} | Resolutions: {resolutions}\n")

    # Time-loop processing
    for t in time_steps:
        t_str = f"{t:.2f}"
        print(f" Processing Time Step: t = {t_str}s")
        
        # 1. Load Reference Conforming Solution
        # Reflecting directory structure: Penalty - RIIS&Brinkman/src/cyl/conforming/moving/n70/
        conf_dir = os.path.join(base_dir, "src", "cyl", "conforming", "moving", "n70", "velocity")
        conf_file = os.path.join(conf_dir, f"velocity_t={t_str}.h5") 
        
        if not os.path.exists(conf_file):
            print(f" [Warning] Conforming reference missing at t={t}. Skipping.")
            continue
            
        u_ex = load_hdf5_solution(conf_file, "velocity")

        # 2. Load Penalty Solutions and compute instantaneous errors
        for method in methods:
            for n in resolutions:
                # Folder match: src/cyl/brinkman/moving/n25_R1000.0/ (or similar)
                # Adjust string formatting based on your exact filename convention
                sim_folder = f"n{n}_R1000.0" 
                chk_dir = os.path.join(base_dir, "src", "cyl", method, "moving", sim_folder, "velocity")
                chk_file = os.path.join(chk_dir, f"velocity_t={t_str}.h5")

                if os.path.exists(chk_file):
                    u_h = load_hdf5_solution(chk_file, "velocity")
                    
                    # Error evaluation
                    err_L2, err_H1 = analyzer.compute_fluid_errors(u_ex, u_h, t)
                    results[method][n]["L2"].append(err_L2)
                    results[method][n]["H1"].append(err_H1)
                    
                    # Store profiles at the final time step for comparison
                    if abs(t - t_end) < 1e-5:
                        y_p, u_p = analyzer.extract_vertical_profile(u_h, t)
                        profiles[method][n] = (y_p, u_p)
                else:
                    print(f" [Warning] Missing penalty file: {chk_file}")
                    # Append NaN to avoid alignment issues if a file is missing
                    results[method][n]["L2"].append(float('nan'))
                    results[method][n]["H1"].append(float('nan'))

        # Reference profile at final time step
        if abs(t - t_end) < 1e-5:
            y_ref, u_ref = analyzer.extract_vertical_profile(u_ex, t)

        print()

    # =============================================================================
    # 3. CONVERGENCE RATE & PLOTTING STAGE
    # =============================================================================
    print("\n>> Computing Global Bochner Norms & Convergence Orders:")

    summary = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"\n--- Method: {method.upper()} ---")
        print(f"{'='*50}")
        # Compute discrete L2(0,T; L2) spatio-temporal error
        errors_L2_T = []
        errors_H1_T = []
        dx_h = [1.0/n for n in resolutions] # Characteristic mesh sizes
        
        for n in resolutions:
            l2_arr = np.array(results[method][n]["L2"])
            h1_arr = np.array(results[method][n]["H1"])

            mask = ~np.isnan(l2_arr)
            if not np.any(mask):
                print(f" Mesh n={n:2d} | No valid data steps found.")
                errors_L2_T.append(float('nan'))
                errors_H1_T.append(float('nan'))
                continue
            
            valid_times = time_steps[mask]
            # Integration over time via Trapezoidal rule (Bochner Norm)
            bochner_L2 = sqrt(np.trapezoid(l2_arr[mask]**2, valid_times))
            bochner_H1 = sqrt(np.trapezoid(h1_arr[mask]**2, valid_times))
            errors_L2_T.append(bochner_L2)
            errors_H1_T.append(bochner_H1)
            
            print(f" Mesh n={n:2d} | Bochner L2 Error: {bochner_L2:.5e} | Bochner H1 Error: {bochner_H1:.5e}")

        print()
        rates_L2 = []
        rates_H1 = []

        for i in range(len(resolutions) - 1):
            e1_L2, e2_L2 = errors_L2_T[i], errors_L2_T[i+1]
            e1_H1, e2_H1 = errors_H1_T[i], errors_H1_T[i+1]

            if not any(np.isnan([e1_L2, e2_L2, e1_H1, e2_H1])):
                log_h = np.log(dx_h[i] / dx_h[i+1])
                p_L2  = np.log(e1_L2 / e2_L2) / log_h
                p_H1  = np.log(e1_H1 / e2_H1) / log_h
                rates_L2.append(p_L2)
                rates_H1.append(p_H1)
                print(f"  Rate n={resolutions[i]}→{resolutions[i+1]}: "
                      f"p_L2={p_L2:+.3f}  p_H1={p_H1:+.3f}")
            else:
                rates_L2.append(float('nan'))
                rates_H1.append(float('nan'))

        summary[method] = {
            "resolutions": resolutions,
            "h":           dx_h,
            "L2":          errors_L2_T,
            "H1":          errors_H1_T,
            "rates_L2":    rates_L2,
            "rates_H1":    rates_H1,
        }

    _save_summary(summary, dirs["data"])
    _save_raw_errors(results, time_steps, resolutions, methods, dirs["data"])

    generate_convergence_plots(summary, dirs["conv_plots"])

    if y_ref is not None:
        generate_profile_plots(y_ref, u_ref, profiles, methods, resolutions, dirs["profiles"])

    print(f"\n>> All the results are saved in: {output_dir}/")

    # Plot final step profile comparison
    # generate_plots(y_ref, u_ref, profiles, methods, resolutions)

def _save_summary(summary, data_dir):
    path = os.path.join(data_dir, "convergence_summary.txt")
    with open(path, "w") as f:
        f.write("CONVERGENCE ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for method, s in summary.items():
            f.write(f"Method: {method.upper()}\n")
            f.write(f"{'n':>5} {'h':>10} {'L2(0,T;L2)':>15} {'L2(0,T;H1)':>15}\n")
            f.write("-" * 50 + "\n")
            for i, n in enumerate(s["resolutions"]):
                f.write(f"{n:>5} {s['h'][i]:>10.4f} {s['L2'][i]:>15.5e} {s['H1'][i]:>15.5e}\n")
            f.write("\nConvergence rates:\n")
            for i, (rL2, rH1) in enumerate(zip(s["rates_L2"], s["rates_H1"])):
                n1, n2 = s["resolutions"][i], s["resolutions"][i+1]
                f.write(f"  n={n1}→{n2}: p_L2={rL2:+.3f}  p_H1={rH1:+.3f}\n")
            f.write("\n")
    print(f"\n   Summary saved: {path}")


def _save_raw_errors(results, time_steps, resolutions, methods, data_dir):
    for method in methods:
        for n in resolutions:
            path = os.path.join(data_dir, f"errors_{method}_n{n}.csv")
            l2 = results[method][n]["L2"]
            h1 = results[method][n]["H1"]
            with open(path, "w") as f:
                f.write("t,L2_error, H1_error\n")
                for t, e2, e1 in zip(time_steps, l2, h1):
                    f.write(f"{t:.2f},{e2:.6e},{e1:.6e}\n")


def generate_convergence_plots(summary, plot_dir):
    """
    Produce due figure:
      1. Grafico log-log errore vs h per L2 e H1
      2. Evoluzione temporale degli errori (se si passa results)
    """
    colors  = {"brinkman": "#2c7bb6", "RIIS": "#d7191c"}
    markers = {25: "o", 50: "s", 70: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Spatial Convergence — Bochner Norms", fontsize=13, fontweight="bold")

    for norm_idx, norm_name in enumerate(["L2", "H1"]):
        ax = axes[norm_idx]
        ax.set_title(f"$L^2(0,T; {'L^2' if norm_name=='L2' else 'H^1'})$ Error")

        for method, s in summary.items():
            h_vals = np.array(s["h"])
            e_vals = np.array(s[norm_name])
            valid  = ~np.isnan(e_vals)
            if valid.sum() < 2:
                continue
            ax.loglog(h_vals[valid], e_vals[valid],
                      color=colors.get(method, "gray"),
                      marker="o", markersize=7, linewidth=1.8,
                      label=method.upper())

        # Linee di riferimento ordine 1 e 2
        h_ref = np.array([1/25, 1/50, 1/70])
        for order, ls, lbl in [(1, "--", "O(h)"), (2, ":", "O(h²)")]:
            scale = 0.3 if norm_name == "L2" else 5.0
            ax.loglog(h_ref, scale * h_ref**order,
                      color="gray", linestyle=ls, linewidth=1.2, label=lbl)

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


def generate_profile_plots(y_ref, u_ref, profiles, methods, resolutions, plot_dir):
    """
    Profilo verticale della velocità al tempo finale per ogni metodo.
    Un file PNG separato per ciascun metodo.
    """
    colors = {25: "#f4a61d", 50: "#2ca02c", 70: "#d62728"}

    for method in methods:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle(
            f"Velocity Magnitude Profile at $t = T$ — {method.upper()}\n"
            "Vertical Cut Through Cylinder Center",
            fontsize=12, fontweight="bold"
        )

        ax.plot(u_ref, y_ref, color="black", linewidth=2.0,
                linestyle="-", label="Conforming (ref.)", zorder=5)

        for n in resolutions:
            if n in profiles[method]:
                y_p, u_p = profiles[method][n]
                ax.plot(u_p, y_p, color=colors[n], linewidth=1.5,
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

        path = os.path.join(plot_dir, f"velocity_profiles_{method}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved: {path}")


# This is an alternative versione of the plot method that puts both profile-analysis in a
# single figure with subplots
def generate_profile_plots_ver2(y_ref, u_ref, profiles, methods, resolutions, plot_dir):
    """
    Profilo verticale della velocità al tempo finale per ogni metodo.
    Tutti i metodi in una sola figura con subplots per confronto diretto.
    """
    colors = {25: "#f4a61d", 50: "#2ca02c", 70: "#d62728"}

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6), sharey=True)
    if n_methods == 1:
        axes = [axes]

    fig.suptitle("Velocity Magnitude Profile at $t = T$ — Vertical Cut Through Cylinder Center",
                 fontsize=12, fontweight="bold")

    for ax, method in zip(axes, methods):
        ax.set_title(f"{method.upper()}", fontsize=11)
        ax.plot(u_ref, y_ref, color="black", linewidth=2.0,
                linestyle="-", label="Conforming (ref.)", zorder=5)

        for n in resolutions:
            if n in profiles[method]:
                y_p, u_p = profiles[method][n]
                ax.plot(u_p, y_p, color=colors[n], linewidth=1.5,
                        linestyle="--", label=f"n={n}")

        ax.set_xlabel("$\\|\\mathbf{u}\\|$ (velocity magnitude)")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("$y$ (vertical coordinate)")
    plt.tight_layout()

    path = os.path.join(plot_dir, "velocity_profiles_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved: {path}")

"""
def generate_plots(y_ref, u_ref, profiles, methods, resolutions):

    colors = {25: '#f4c430', 50: '#5cb85c', 70: '#d9534f'}
    
    for method in methods:
        plt.figure(figsize=(9, 6))
        plt.title(f"2D Flow Around Moving Cylinder - {method.upper()} Spatial Convergence", fontsize=12, pad=15)
        
        # Plot exact reference
        plt.plot(y_ref, u_ref, label='Conforming (Reference)', color='black', linewidth=1.75, linestyle='-')
        
        # Plot penalized approximations
        for n in resolutions:
            if n in profiles[method]:
                plt.plot(y_ref, profiles[method][n], label=f'{method.capitalize()} n={n}', 
                         color=colors[n], linewidth=1.5, linestyle='--')
        
        # Post-styling
        plt.xlim(0.0, 1.0)
        plt.xticks(np.arange(0.0, 1.05, 0.05), rotation=45)
        plt.xlabel("Vertical Coordinate (y)", fontsize=10)
        plt.ylabel("Velocity Magnitude ||u||", fontsize=10)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', framealpha=1.0, edgecolor='black', fontsize=9)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        
        plt.savefig(f"convergence_profile_{method}.png", dpi=300)
        plt.show()
"""

# =============================================================================
# 4. EXECUTION TARGET
# =============================================================================
if __name__ == "__main__":
    # 1. Import your real obstacle class and domain settings
    # Since 'src' and 'domain_settings' are both under 'Penalty - RIIS&Brinkman', 
    # we need to ensure Python can find 'domain_settings' by moving up one level.
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    parent_dir = os.path.dirname(current_dir) 
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        
    from domain_settings.obstacles import circleObstacle

    # 2. Instantiate your obstacle with the physical parameters 
    obstacle = circleObstacle(x=0.5, y=0.5, r=0.1) 

    # 3. Define the project directory path 
    PROJECT_DIR = parent_dir
    OUTPUT_DIR  = os.path.join(current_dir, "results")
    
    # 4. Trigger the full automated pipeline
    run_pipeline(PROJECT_DIR, obstacle, output_dir=OUTPUT_DIR)