# Firedrake non implementa FFC_parameters globali allo stesso modo di dolfin.
# L'ottimizzazione del codice generato C (UFLACS/TSFC) è attiva di default.

# Mappatura completa delle opzioni di algebra lineare su dizionari PETSc
krylov_solvers = {
    "ksp_monitor": False,
    "ksp_view": False,
    "ksp_converged_reason": False,
    "ksp_initial_guess_nonzero": True,
    "ksp_max_it": 300,
    "ksp_atol": 1e-8
}

tentative_velocity_solver = dict(
    solver_type='bicgstab',
    preconditioner_type='jacobi')

pressure_correction_solver = dict(
    solver_type='gmres',
    preconditioner_type='hypre') # 'hypre_amg' diventa 'hypre' in PETSc/Firedrake

velocity_correction_solver = dict(
    solver_type='cg',
    preconditioner_type='jacobi')

energy_conservation_solver = dict(
    solver_type='bicgstab',
    preconditioner_type='jacobi')

pressure_velocity_coupling = "IPCS"                 
piso_iterations = 10                                 
piso_tol = 1e-3                                     

custom_newtons_solver = True
line_search_solver = False

solid_momentum_solver = dict(solver_type='bicgstab')                 

if not problem_physics['compressible_solid']:
    solid_momentum_solver.update(solver_type='mumps')               
    custom_newtons_solver = False

if custom_newtons_solver:
    solid_momentum_solver.update(solver_type='bcgs')

# Parametri del risolutore non lineare (NonlinearVariationalSolver) in Firedrake
solid_displacement_parameters = {
    "snes_type": "newtonls",
    "ksp_type": solid_momentum_solver['solver_type'],
    "pc_type": "hypre",
    "snes_monitor": True,
    "snes_atol": 1e-15,
    "snes_rtol": 1e-6,
    "snes_max_it": 20
}

solid_displacement_custom_solver_parameters = {
    "snes_atol": 1e-15,
    "snes_rtol": 1e-6,
    "snes_max_it": 20,
    "snes_monitor": True
}

snes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_monitor": True,
    "ksp_type": "bicgstab",
    "snes_atol": 1e-9,
    "snes_rtol": 1e-7,
    "snes_max_it": 10
}