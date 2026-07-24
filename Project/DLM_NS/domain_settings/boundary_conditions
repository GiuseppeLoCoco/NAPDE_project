from firedrake import DirichletBC, Constant, SpatialCoordinate, sin, pi, as_vector

t_param = Constant(0.0)

def get_inflow_profile(mesh):
    X = SpatialCoordinate(mesh)
    return 6.0 * X[1] * (4.1 - X[1]) / (4.1 * 4.1) * sin(2.0 * pi * t_param / 10.0)

def time_varying_bc(tt):
    t_param.assign(tt)

def create_boundary_conditions(fluid_mesh, **V):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh)
    # Target velocity sub-space (V['fluid'][0]) and pressure sub-space (V['fluid'][1])
    bcu_inflow = DirichletBC(V['fluid'][0], as_vector([inflow_profile, 0.0]), 3)
    bcu_walls = DirichletBC(V['fluid'][0], Constant((0.0, 0.0)), (1, 2))
    bcp_outflow = DirichletBC(V['fluid'][1], Constant(0.0), 4)

    bcs = [bcu_inflow, bcu_walls, bcp_outflow]
    return bcs
