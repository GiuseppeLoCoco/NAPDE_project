
tentative_velocity_solver = dict(
    solver_type='bicgstab',
    preconditioner_type='jacobi')


velocity_correction_solver = dict(
    solver_type='cg',
    preconditioner_type='jacobi')


# Timing options: set to True to print execution time for each iteration / time step
print_iteration_time = True

# Resume options: if True, solvers will automatically resume from the latest available checkpoint
resume_simulation = True

