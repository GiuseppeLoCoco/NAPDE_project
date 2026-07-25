from firedrake import DirichletBC, Constant, SpatialCoordinate, sin, pi, as_vector, exp

t_param = Constant(0.0)

def get_inflow_profile(mesh):
    X = SpatialCoordinate(mesh)
    y = X[1]
    # Profilo di inflow parabolico con fase di "start-up" per un canale di altezza 1.0
    inflow_x = (1.0 - exp(-t_param)) * 4.0 * y * (1.0 - y)
    return as_vector([inflow_x, 0.0])

def time_varying_bc(tt):
    t_param.assign(tt)

def create_boundary_conditions(fluid_mesh, **V):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh)
    # Target velocity sub-space (V['fluid'][0]) and pressure sub-space (V['fluid'][1])
    bcu_inflow = DirichletBC(V['fluid'][0], inflow_profile, 3)
    bcu_walls = DirichletBC(V['fluid'][0], Constant((0.0, 0.0)), (1, 2))
    bcp_outflow = DirichletBC(V['fluid'][1], Constant(0.0), 4) # Impone p=0 all'outflow (ID 4)

    bcs = [bcu_inflow, bcu_walls, bcp_outflow]
    return bcs

def create_brinkman_riis_bcs(W, mesh):
    """
    Crea le condizioni al contorno per i solutori Brinkman e RIIS
    che utilizzano una RectangleMesh standard.
    IDs: 1 (inflow), 2 (outflow), 3 (bottom wall), 4 (top wall).
    """
    inflow_profile = get_inflow_profile(mesh)

    inflow_id = 1
    outflow_id = 2
    walls_ids = (3, 4)

    bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
    bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow_id) # Impone p=0 all'outflow (ID 2)
    
    return [bcu_inflow, bcu_walls, bcp_outflow]
