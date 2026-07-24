
tentative_velocity_solver = dict(
    solver_type='bicgstab',
    preconditioner_type='jacobi')


velocity_correction_solver = dict(
    solver_type='cg',
    preconditioner_type='jacobi')

