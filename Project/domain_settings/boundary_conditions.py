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

def create_boundary_conditions(fluid_mesh, type_obstacle: str, structured: bool = False, **V):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh, type_obstacle=type_obstacle)
    inflow_id = 3 if structured else 1
    outflow_id = 4 if structured else 2
    walls_ids = (1, 2) if structured else (3, 4)

    bcu_inflow = DirichletBC(V['fluid'][0], inflow_profile, inflow_id)
    bcu_walls = DirichletBC(V['fluid'][0], Constant((0.0, 0.0)), walls_ids)
    bcp_outflow = DirichletBC(V['fluid'][1], Constant(0.0), outflow_id)

    bcs = [bcu_inflow, bcu_walls, bcp_outflow]
    return bcs

def create_boundary_conditions_correction(fluid_mesh, V, type_obstacle: str, structured: bool = False):
    inflow_profile = get_inflow_profile(fluid_mesh.mesh, type_obstacle=type_obstacle)
    inflow_id = 3 if structured else 1
    walls_ids = (1, 2) if structured else (3, 4)

    bcu_inflow = DirichletBC(V, inflow_profile, inflow_id)
    bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), walls_ids)

    bcs = [bcu_inflow, bcu_walls]
    return bcs

def create_bcs_penalty(W, mesh, type_obstacle: str, structured: bool = True):
    """
    Crea le condizioni al contorno per i solutori Brinkman e RIIS.
    Structured RectangleMesh: 3 (inflow), 4 (outflow), (1, 2) (bottom/top walls).
    Gmsh mesh: 1 (inflow), 2 (outflow), (3, 4) (bottom/top walls).
    """
    inflow_profile = get_inflow_profile(mesh, type_obstacle=type_obstacle)

    inflow_id = 3 if structured else 1
    outflow_id = 4 if structured else 2
    walls_ids = (1, 2) if structured else (3, 4)

    bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow_id)
    bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls_ids)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow_id)
    
    return [bcu_inflow, bcu_walls, bcp_outflow]

def create_bcs_conforming(W, mesh, w_obstacle_velocity, type_obstacle: str, structured: bool = False):
    """
    Crea le condizioni al contorno per il solutore Conforming.
    """
    inflow_profile = get_inflow_profile(mesh, type_obstacle=type_obstacle)
    inflow_id = 3 if structured else 1
    walls_ids = (1, 2) if structured else (3, 4)

    bcs = [
        DirichletBC(W.sub(0), inflow_profile, inflow_id),
        DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[0]),
        DirichletBC(W.sub(0), Constant((0, 0)), walls_ids[1])
    ]
    if w_obstacle_velocity is not None:
        bcs.append(DirichletBC(W.sub(0), w_obstacle_velocity, 5))
    return bcs
