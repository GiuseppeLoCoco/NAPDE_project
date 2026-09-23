# NAPDE_project

> *"Moving geometries without remeshing: assessment and comparison of immersed methods for CFD"*

This repository contains a comprehensive finite element framework for simulating fluid flows around static and moving obstacles using **immersed boundary and fictitious domain methods** without body-fitted remeshing, implemented in [Firedrake](https://www.firedrakeproject.org/).

The project benchmarks and compares three immersed approaches against standard conforming (body-fitted) Navier-Stokes solutions:
1. **$L^2$ Brinkman Volume Penalization** (`NS_Brinkman.py`)
2. **Regularized Immersed Interface Solver (RIIS)** (`NS_RIIS.py`)
3. **Distributed Lagrange Multiplier / Fictitious Domain (DLM/FD)** (`NS_DLM_simple.py`)
4. **Body-Fitted Conforming Solver** (`NS_Conforming.py`) — reference baseline

---

## Table of Contents

- [Prerequisites & Container Setup](#prerequisites--container-setup)
- [Repository Structure](#repository-structure)
- [Running the Solvers](#running-the-solvers)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Editing Code Directly](#editing-code-directly)
- [Running Validation & Benchmark Scripts](#running-validation--benchmark-scripts)
  - [L2 Penalization Benchmark (Angot et al. 1999)](#l2-penalization-benchmark-angot-et-al-1999)
  - [Convergence Analysis](#convergence-analysis)
- [Running Buffer Recovery Experiments](#running-buffer-recovery-experiments)
- [Results & Output Files](#results--output-files)

---

## Prerequisites & Container Setup

Due to specific dependencies (Firedrake, PETSc, Gmsh, h5py, libspatialindex, MPI), all simulations should be executed inside the official pre-configured container.

### 1. Download the Container
Download the container image (`.sif` file) from the following link:
- **[Container Firedrake](https://polimi365-my.sharepoint.com/:f:/g/personal/10860214_polimi_it/IgCUXJ80Dh0iQ5lFfHaQzq0NAdDCe_60vJw6Kzce2Xfz6Xk?e=vrTBbD)**

### 2. Running with Apptainer / Singularity

You can execute scripts either by entering an interactive shell inside the container or by executing commands directly from the host terminal.

#### Option A: Interactive Shell (Recommended)
Open an interactive session inside the container:
```bash
apptainer shell path/to/firedrake-vanilla-default.sif
# (or: singularity shell path/to/firedrake-vanilla-default.sif)
```
Once inside the container shell, navigate to the repository and run any Python script directly:
```bash
cd Project/Solvers
python3 NS_Brinkman.py --obstacle cylinder --dt 0.1 --t_final 10.0
```

#### Option B: Direct Execution from Host
Run commands through `apptainer exec` without entering an interactive shell:
```bash
apptainer exec path/to/firedrake-container.sif python3 Project/Solvers/NS_Brinkman.py --obstacle cylinder
```

---

## Repository Structure

The core codebase is located inside the `Project/` directory, organized as follows:

```text
Project/
├── Solvers/                   # Navier-Stokes & Stokes solver implementations
│   ├── NS_Conforming.py       # Conforming (body-fitted mesh) solver (benchmark reference)
│   ├── NS_Brinkman.py         # Brinkman L2 volume penalization solver
│   ├── NS_RIIS.py             # Regularized Immersed Interface Solver (RIIS)
│   ├── NS_DLM_simple.py       # Distributed Lagrange Multiplier (DLM/FD) solver
│   └── Stokes_solver.py       # Steady Stokes solver (used for initial conditions)
│
├── domain_settings/           # Mesh generation, obstacles, and boundary conditions
│   ├── mesh_settings.py       # Structured/unstructured mesh generation (Cartesian & Gmsh)
│   ├── obstacles.py           # Solid geometry definitions (square, cylinder, line, rotating)
│   ├── boundary_conditions.py # Inflow profiles, no-slip walls, outlet, and penalty conditions
│   └── delta_interpolation.py # Regularized Dirac delta kernel interpolation for DLM coupling
│
├── user_inputs/               # Global simulation parameters and solver configurations
│   ├── user_parameters.py     # Domain size, obstacle parameters, Re, R, FE polynomial degrees
│   └── solver_options.py      # PETSc linear solver/preconditioner options, checkpoint & timing flags
│
├── Utils/                     # Utilities and post-processing tools
│   ├── post_processing.py     # Checkpoint save/load, ParaView PVD/VTK export, drag/lift computation
│   ├── experiment_plots.py    # Plotting routines for buffer layer recovery and velocity profiles
│   └── mms.py                 # Method of Manufactured Solutions (MMS) exact verification
│
├── validation/                # Convergence analysis and scientific benchmark scripts
│   ├── test_L2_penalization.py# L2 Brinkman penalization benchmark (Angot et al. 1999)
│   ├── convergence_analysis.py# Grid refinement convergence analysis (L2, H1 velocity & L2 pressure)
│   ├── checkpoint_loader.py   # Checkpoint inspection and HDF5 solution retrieval utilities
│   ├── validation_plots.py    # Error plotting and flow contour visualization
│   └── Experiments/           # Buffer layer recovery benchmark experiments
│       ├── test_Brinkman.py   # Buffer recovery analysis for Brinkman penalization
│       ├── test_RIIS.py       # Buffer recovery analysis for RIIS
│       ├── test_DLM.py        # Buffer recovery analysis for DLM/FD
│     
│
└── Plots/                     # Generated figures, velocity profile comparisons, and convergence curves
```

---

## Running the Solvers

You can run each solver in two ways:
1. **Via command-line arguments (CLI)** from the terminal.
2. **By modifying parameters directly in the script** (`__main__` entrypoint or `user_parameters.py`).

### Command-Line Interface (CLI)

Each solver script provides a command-line interface via `argparse`. The following arguments are supported:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--obstacle` | `str` | `'cylinder'` | Type of obstacle: `cylinder`, `square`, `line`, `rotating`, `rotating_line` |
| `--moving` | flag | `True` | Enable obstacle motion / kinematics |
| `--dt` | `float` | Solver-dependent | Time step size $\Delta t$ |
| `--t_final` | `float` | Solver-dependent | Final simulation physical time $T$ |
| `--resume` | flag | `True` | Resume simulation from the latest checkpoint if available |
| `--restart` | flag | — | Restart simulation from $t=0$, ignoring checkpoints |
| `--print_time` / `--no_print_time` | flag | `None` | Toggle execution time printout per time step |

#### Examples:

- **Conforming Solver** (Body-fitted mesh reference):
  ```bash
  cd Project/Solvers
  python3 NS_Conforming.py --obstacle cylinder --dt 0.05 --t_final 10.0
  ```

- **Brinkman Volume Penalization**:
  ```bash
  cd Project/Solvers
  python3 NS_Brinkman.py --obstacle cylinder --dt 0.1 --t_final 20.0
  ```

- **Regularized Immersed Interface Solver (RIIS)**:
  ```bash
  cd Project/Solvers
  python3 NS_RIIS.py --obstacle square --dt 0.1 --t_final 10.0
  ```

- **Distributed Lagrange Multiplier (DLM/FD)**:
  ```bash
  cd Project/Solvers
  python3 NS_DLM_simple.py --obstacle cylinder --dt 0.1 --t_final 20.0
  ```

- **Restarting a simulation from scratch (ignoring checkpoints)**:
  ```bash
  python3 NS_Brinkman.py --obstacle square --restart
  ```

### Editing Code Directly

If you prefer modifying code directly rather than using terminal arguments:
1. Open the solver file (e.g. `Project/Solvers/NS_Brinkman.py`).
2. Scroll to the bottom `if __name__ == '__main__':` block.
3. Edit the default parameters in the `solver.Brinkman_solve(...)` or initialization call:
   ```python
   solver = Brinkman_solver(moving=False, type_obstacle='square', print_iteration_time=True)
   solver.Brinkman_solve(dt=0.05, t_final=15.0, resume=False)
   ```
4. For global settings (mesh resolution `n`, Reynolds number `Re`, penalty parameter `R`, channel dimensions `Lx`, `Ly`), modify `Project/user_inputs/user_parameters.py`.

---

## Running Validation & Benchmark Scripts

### L2 Penalization Benchmark (Angot et al. 1999)

The script `Project/validation/test_L2_penalization.py` verifies the Brinkman $L^2$ penalization method against variations of the permeability / penalty parameter $\eta = 1/R$. It assesses velocity field attenuation inside the solid obstacle and boundary approximation error in conformity with the benchmark study:

> **P. Angot, C.-H. Bruneau, and P. Fabrie.**  
> *A penalization method to take into account obstacles in incompressible viscous flows.*  
> Numerische Mathematik, 81(4): 497–520, 1999.  
> DOI: [10.1007/s002110050401](https://doi.org/10.1007/s002110050401)

The test supports two regimes:
- **Steady regime** ($Re = 40$): Tests permeability values $\eta \in [10^{-2}, 10^{-6}]$.
- **Unsteady regime** ($Re = 80$): Tests permeability values $\eta \in [10^{-2}, 10^{-8}]$.

#### Running from Terminal:
```bash
cd Project/validation

# Run steady flow benchmark (Re = 40)
python3 test_L2_penalization.py --mode steady

# Run unsteady flow benchmark (Re = 80)
python3 test_L2_penalization.py --mode unsteady

# Run both regimes
python3 test_L2_penalization.py --mode all
```
### Convergence Analysis

The script `Project/validation/convergence_analysis.py` evaluates the spatial convergence order of the immersed methods against a high-resolution conforming simulation ($n = 320$). It computes:
- Relative velocity $L^2$ error: $\|u_{immersed} - u_{conforming}\|_{L^2(\Omega_f)} / \|u_{conforming}\|_{L^2(\Omega_f)}$
- Relative velocity $H^1$ semi-norm error: $|u_{immersed} - u_{conforming}|_{H^1(\Omega_f)} / |u_{conforming}|_{H^1(\Omega_f)}$
- Relative pressure $L^2$ error: $\|p_{immersed} - p_{conforming}\|_{L^2(\Omega_f)} / \|p_{conforming}\|_{L^2(\Omega_f)}$
- Empirical convergence rates and automated log-log plots.

#### Running from Terminal:
```bash
cd Project/validation
python3 convergence_analysis.py
`
---

## Running Buffer Recovery Experiments

The directory `Project/validation/Experiments/` contains scripts testing the **upstream buffer layer recovery mechanism**. In these experiments:
1. **Phase 1**: A reference simulation on a shortened channel $\Omega_0$ is conducted.
2. **Phase 2**: An upstream buffer region $\Omega_{buf}$ is added where the flow is forced/penalized to assess how accurately the immersed boundary method recovers the fully developed parabolic profile downstream across different grid resolutions.

### Available Experiment Scripts:

1. **`test_Brinkman.py`**: Buffer recovery using the $L^2$ Brinkman penalization method.
2. **`test_RIIS.py`**: Buffer recovery using the Regularized Immersed Interface Solver.
3. **`test_DLM.py`**: Buffer recovery using the Distributed Lagrange Multiplier (DLM/FD) formulation.
4. **`test_Brinkman_ResistiveTerm.py`**: Sensitivity and scaling analysis of the Brinkman resistive parameter $R$ in the buffer zone.

### Running from Terminal:

```bash
cd Project/validation/Experiments

# Brinkman buffer recovery test
python3 test_Brinkman.py

# RIIS buffer recovery test
python3 test_RIIS.py

# DLM/FD buffer recovery test
python3 test_DLM.py

# Resistive term scaling test
python3 test_Brinkman_ResistiveTerm.py
```

### Modifying Parameters in the Script:
Parameters can be configured directly inside the `if __name__ == "__main__":` block of each experiment file:
```python

---

## Results & Output Files

- **Visualizations (ParaView)**: Simulation states are saved as `.pvd` and `.vtu` files in subdirectories within `Project/Solvers/` and `Project/validation/Experiments/`.
- **Checkpoints**: Simulation checkpoints are stored in HDF5 (`.h5`) format, enabling seamless resumption and error evaluation without re-running long simulations.
- **Plots & Metrics**: Convergence error curves, profile cuts, and benchmark comparisons are exported to `Project/Plots/` or specific experiment result folders (e.g. `results_Brinkman_buffer_recovery_unstructured/`, `results_RIIS_buffer_recovery/`, `results_dlm_buffer_recovery/`).

