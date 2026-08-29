from firedrake import DirichletBC, Constant, SpatialCoordinate, sin, pi, as_vector, exp

t_param = Constant(0.0)

def get_inflow_profile(mesh, type_obstacle):
    X = SpatialCoordinate(mesh)
    y = X[1]

    if type_obstacle == "cylinder":
        # Parabolic inflow profile with "start-up" phase for a channel with height h = 1.0
        inflow_x = (1.5) * (1.0 - exp(-t_param)) * 4.0 * y * (1.0 - y)
        return as_vector([inflow_x, 0.0])
    elif type_obstacle == "square":
        # Parabolic inflow profile without "start-up" phase for a channel with height h = 1.0
        inflow_x = (1.5) * 4.0 * y * (1.0 - y)
        return as_vector([inflow_x, 0.0])
    elif type_obstacle in ["line", "rotating", "rotating_line"]:
        # Parabolic inflow profile with "start-up" phase
        inflow_x = (1.5) * (1.0 - exp(-t_param)) * 4.0 * y * (1.0 - y)
        return as_vector([inflow_x, 0.0])
    else:
        # Default parabolic profile
        inflow_x = (1.5) * 4.0 * y * (1.0 - y)
        return as_vector([inflow_x, 0.0])

def time_varying_bc(tt):
    t_param.assign(tt)

def create_boundary_conditions(fluid_mesh, type_obstacle: str, **V):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh, type_obstacle = type_obstacle)
    # Target velocity sub-space (V['fluid'][0]) and pressure sub-space (V['fluid'][1])
    bcu_inflow = DirichletBC(V['fluid'][0], inflow_profile, 1)
    bcu_walls = DirichletBC(V['fluid'][0], Constant((0.0, 0.0)), (3, 4))
    bcp_outflow = DirichletBC(V['fluid'][1], Constant(0.0), 2) # Null pressure at the outflow

    bcs = [bcu_inflow, bcu_walls, bcp_outflow]
    return bcs

def create_boundary_conditions_correction(fluid_mesh, V, type_obstacle: str):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh, type_obstacle = type_obstacle)
    # Target velocity sub-space (V['fluid'][0]) and pressure sub-space (V['fluid'][1])
    bcu_inflow = DirichletBC(V, inflow_profile, 1)
    bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), (3, 4))

    bcs = [bcu_inflow, bcu_walls]
    return bcs

def create_bcs_penalty(W, mesh, type_obstacle: str):
    """
    Crea le condizioni al contorno per i solutori Brinkman e RIIS
    che utilizzano una RectangleMesh standard.
    IDs: 1 (inflow), 2 (outflow), 3 (bottom wall), 4 (top wall).
    """
    inflow_profile = get_inflow_profile(mesh, type_obstacle = type_obstacle)

    inflow_id = 1
    outflow_id = 2
    walls_ids = (3, 4)

    bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
    bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow_id) # Null pressure at the outflow
    
    return [bcu_inflow, bcu_walls, bcp_outflow]

def create_bcs_conforming(W, mesh, w_obstacle_velocity, type_obstacle: str): # t_constant is now handled by t_param global
    """
    Crea le condizioni al contorno per il solutore Conforming.
    IDs: 1 (inflow), 3 (bottom wall), 4 (top wall), 5 (obstacle).
    Non impone condizioni di Dirichlet sulla pressione all'outflow.
    """
    # Usa la funzione condivisa per il profilo di inflow
    inflow_profile = get_inflow_profile(mesh, type_obstacle = type_obstacle)

    # Define boundaries (IDs are consistent with mesh_settings.py)
    return [DirichletBC(W.sub(0), inflow_profile, 1), # Inflow
            DirichletBC(W.sub(0), Constant((0, 0)), 3), # Bottom wall
            DirichletBC(W.sub(0), Constant((0, 0)), 4), # Top wall
            DirichletBC(W.sub(0), w_obstacle_velocity, 5)] # Obstacle
