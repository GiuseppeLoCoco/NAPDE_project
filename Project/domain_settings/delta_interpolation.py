from firedrake import Function

class FSIInterpolation:
    def __init__(self):
        self.fluid_space = None
        self.solid_space = None
        self.h = None

    
    def extract_dof_component_map_user(self, space, flag):
        if flag == "F":
            self.fluid_space = space
        elif flag == "S":
            self.solid_space = space


def interpolate_nonmatching_mesh_delta(fsi_interpolation, source_func, mesh_flag):
    """
    Interpolates a Firedrake Function `source_func` onto the target function space
    associated with `mesh_flag` ("F" for fluid mesh, "S" for solid mesh) using Firedrake's
    native cross-mesh interpolation with `allow_missing_dofs=True`.
    """
    if mesh_flag == "F":
        target_space = fsi_interpolation.fluid_space
    elif mesh_flag == "S":
        target_space = fsi_interpolation.solid_space
    else:
        raise ValueError(f"Unknown mesh_flag '{mesh_flag}'. Expected 'F' or 'S'.")

    if target_space is None:
        raise RuntimeError(f"Target space for mesh_flag '{mesh_flag}' has not been set in FSIInterpolation.")

    res = Function(target_space)
    res.interpolate(source_func, allow_missing_dofs=True)
    return res

def interpolate_nonmatching_mesh(source_func, target_space):
    """
    Interpolates a Firedrake Function `source_func` onto a `target_space`.
    This is a general-purpose non-matching mesh interpolation.
    """
    res = Function(target_space)
    res.interpolate(source_func, allow_missing_dofs=True)
    return res


# Alias for compatibility with typos in caller scripts
interpolate_nonmetching_mesh_delta = interpolate_nonmatching_mesh_delta
