from firedrake import Mesh, RectangleMesh
import numpy as np
import os, csv, operator
from os import listdir, path, makedirs
import gmsh # Keep gmsh import here for the Mesh(model) constructor and other gmsh calls

from user_inputs import *
from domain_settings import *
from .obstacles import circleObstacle, squareObstacle, rotatingLineObstacle, lineObstacle # Import obstacle types

"""
def conforming_mesh(Lx, Ly, obstacle_obj, n, t_val=0.0):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Cylinder")

    # Resolution
    res = Ly / n

    # Geometry
    channel = model.occ.addRectangle(0, 0, 0, Lx, Ly)
    
    if isinstance(obstacle_obj, circleObstacle):
        cylinder = model.occ.addDisk(obstacle_obj.x_obs, obstacle_obj.y_obs, 0, obstacle_obj.r, obstacle_obj.r)

    elif isinstance(obstacle_obj, squareObstacle):
        xmin = obstacle_obj.x_obs - obstacle_obj.half_side
        ymin = obstacle_obj.y_obs - obstacle_obj.half_side
        # Gmsh addRectangle takes x, y, z, dx, dy
        cylinder = model.occ.addRectangle(xmin, ymin, 0, obstacle_obj.side_length, obstacle_obj.side_length)

    elif isinstance(obstacle_obj, rotatingLineObstacle):
        # Get parameters for the unrotated thin rectangle and rotation
        # We generate the mesh for the orientation at the given t_val
        xmin, ymin, dx, dy, rot_cx, rot_cy, angle = obstacle_obj.get_gmsh_rectangle_params(t_val)
        
        # Create the unrotated rectangle
        unrotated_rect = model.occ.addRectangle(xmin, ymin, 0, dx, dy)
        
        # Apply rotation if the initial angle is not zero
        if abs(angle) > 1e-9:
            model.occ.rotate([(2, unrotated_rect)], rot_cx, rot_cy, 0, 0, 0, 1, angle)
        
        cylinder = unrotated_rect # The (potentially rotated) rectangle is now the 'cylinder' for subtraction
    else:
        raise ValueError(f"Unsupported obstacle type '{type(obstacle_obj)}' for conforming mesh.")


    # Subtraction (Fluid = Channel - Cylinder)
    fluid_shape, _ = model.occ.cut([(2, channel)], [(2, cylinder)])
    model.occ.synchronize()

    # --- TAGGING ---
    # Fluid surface
    model.addPhysicalGroup(2, [fluid_shape[0][1]], name="Fluid")

    # Borders (Lines)
    lines = model.getEntities(1)
    for line in lines:
        mass_center = model.occ.getCenterOfMass(line[0], line[1])
        # x=0: Inflow, x=Lx: Outflow, y=0: Bottom, y=Ly: Top, else: Cylinder
        if np.isclose(mass_center[0], 0): model.addPhysicalGroup(1, [line[1]], 1)    # Inflow
        elif np.isclose(mass_center[0], Lx): model.addPhysicalGroup(1, [line[1]], 2) # Outflow
        elif np.isclose(mass_center[1], 0): model.addPhysicalGroup(1, [line[1]], 3)  # Bottom wall
        elif np.isclose(mass_center[1], Ly): model.addPhysicalGroup(1, [line[1]], 4) # Top wall
        else: model.addPhysicalGroup(1, [line[1]], 5)                                # Cylinder

    model.mesh.setSize(model.getEntities(0), res)
    model.mesh.generate(2)

    m = Mesh(model)
    gmsh.finalize()
    return m
"""


def structured_conforming_square_mesh(Lx, Ly, obstacle_obj, n, t_val=0.0):
    """
    Creates a structured (transfinite) conforming mesh around a square obstacle.
    The domain [0, Lx] x [0, Ly] is subdivided into 8 rectangular blocks surrounding
    the square obstacle, with perfectly aligned Cartesian grid spacing h = Lx / n.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add(f"StructuredConformingSquare_t_{t_val:.4f}")

    h = Lx / float(n)

    dx = float(obstacle_obj.displ_x(t_val)) if hasattr(obstacle_obj, 'displ_x') else 0.0
    dy = float(obstacle_obj.displ_y(t_val)) if hasattr(obstacle_obj, 'displ_y') else 0.0
    xc = obstacle_obj.x_obs + dx if hasattr(obstacle_obj, 'x_obs') else dx
    yc = obstacle_obj.y_obs + dy if hasattr(obstacle_obj, 'y_obs') else dy

    half_side = obstacle_obj.half_side if hasattr(obstacle_obj, 'half_side') else (obstacle_obj.side_length / 2.0)
    x1, x2 = xc - half_side, xc + half_side
    y1, y2 = yc - half_side, yc + half_side

    # 8 rectangular blocks around the square obstacle
    blocks = [
        (0.0, 0.0, x1, y1),               # Bottom-Left
        (x1, 0.0, x2 - x1, y1),           # Bottom-Center
        (x2, 0.0, Lx - x2, y1),           # Bottom-Right
        (0.0, y1, x1, y2 - y1),           # Middle-Left
        (x2, y1, Lx - x2, y2 - y1),       # Middle-Right
        (0.0, y2, x1, Ly - y2),           # Top-Left
        (x1, y2, x2 - x1, Ly - y2),       # Top-Center
        (x2, y2, Lx - x2, Ly - y2),       # Top-Right
    ]

    rect_tags = []
    for x, y, dx_b, dy_b in blocks:
        r = model.occ.addRectangle(x, y, 0, dx_b, dy_b)
        rect_tags.append((2, r))

    # Fragment to ensure all adjacent interfaces share curves and nodes
    out_dim_tags, _ = model.occ.fragment(rect_tags, [])
    model.occ.synchronize()

    # Fluid physical surface (all 8 blocks)
    surfaces = [tag for dim, tag in out_dim_tags if dim == 2]
    model.addPhysicalGroup(2, surfaces, name="Fluid")

    # Set transfinite curves with exact cell count matching spacing h
    curves = model.getEntities(1)
    tol = 1e-5
    for dim, c_tag in curves:
        bbox = model.occ.getBoundingBox(dim, c_tag)
        dx_c = abs(bbox[3] - bbox[0])
        dy_c = abs(bbox[4] - bbox[1])
        length = max(dx_c, dy_c)
        num_cells = max(1, int(round(length / h)))
        model.mesh.setTransfiniteCurve(c_tag, num_cells + 1)

    for s_tag in surfaces:
        model.mesh.setTransfiniteSurface(s_tag, arrangement="Left")

    # Boundary identification and tagging
    inflow_lines, outflow_lines = [], []
    bottom_lines, top_lines = [], []
    obstacle_lines = []

    for dim, line_tag in model.getEntities(1):
        com = model.occ.getCenterOfMass(dim, line_tag)
        xc_l, yc_l = com[0], com[1]

        if np.isclose(xc_l, 0.0, atol=tol):
            inflow_lines.append(line_tag)
        elif np.isclose(xc_l, Lx, atol=tol):
            outflow_lines.append(line_tag)
        elif np.isclose(yc_l, 0.0, atol=tol):
            bottom_lines.append(line_tag)
        elif np.isclose(yc_l, Ly, atol=tol):
            top_lines.append(line_tag)
        elif (x1 - tol <= xc_l <= x2 + tol) and (y1 - tol <= yc_l <= y2 + tol):
            obstacle_lines.append(line_tag)

    if inflow_lines: model.addPhysicalGroup(1, inflow_lines, 1, name="Inflow")
    if outflow_lines: model.addPhysicalGroup(1, outflow_lines, 2, name="Outflow")
    if bottom_lines: model.addPhysicalGroup(1, bottom_lines, 3, name="Bottom")
    if top_lines: model.addPhysicalGroup(1, top_lines, 4, name="Top")
    if obstacle_lines: model.addPhysicalGroup(1, obstacle_lines, 5, name="Obstacle")

    model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    import os
    tmp_msh_file = f"tmp_structured_mesh_{os.getpid()}_{t_val:.4f}.msh"
    gmsh.write(tmp_msh_file)
    gmsh.finalize()

    m = Mesh(tmp_msh_file)
    if os.path.exists(tmp_msh_file):
        os.remove(tmp_msh_file)

    return m


def conforming_mesh(Lx, Ly, obstacle_obj, n, t_val=0.0, structured=False):
    obs_type_name = type(obstacle_obj).__name__
    is_square = (obs_type_name == "squareObstacle" or isinstance(obstacle_obj, squareObstacle))

    if structured and is_square:
        return structured_conforming_square_mesh(Lx, Ly, obstacle_obj, n, t_val=t_val)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    model = gmsh.model
    model.add(f"FluidDomain_t_{t_val:.4f}")

    # Channel resolution based on n
    res_domain = Ly / n
    # Refined resolution at the obstacle boundary
    res_obstacle = res_domain / 2.0  

    # Creation of the external channel
    channel = model.occ.addRectangle(0, 0, 0, Lx, Ly)
    
    # Compute position and geometry of the obstacle at time t_val
    # We obtain the instantaneous displacement (numerical value)
    dx = float(obstacle_obj.displ_x(t_val)) if hasattr(obstacle_obj, 'displ_x') else 0.0
    dy = float(obstacle_obj.displ_y(t_val)) if hasattr(obstacle_obj, 'displ_y') else 0.0
    
    xc = obstacle_obj.x_obs + dx if hasattr(obstacle_obj, 'x_obs') else dx
    yc = obstacle_obj.y_obs + dy if hasattr(obstacle_obj, 'y_obs') else dy

    if obs_type_name == "circleObstacle" or isinstance(obstacle_obj, circleObstacle):
        obstacle = model.occ.addDisk(xc, yc, 0, obstacle_obj.r, obstacle_obj.r)

    elif is_square:
        xmin = xc - obstacle_obj.half_side
        ymin = yc - obstacle_obj.half_side
        obstacle = model.occ.addRectangle(xmin, ymin, 0, obstacle_obj.side_length, obstacle_obj.side_length)

    elif obs_type_name in ["rotatingLineObstacle", "lineObstacle"] or isinstance(obstacle_obj, lineObstacle) or isinstance(obstacle_obj, rotatingLineObstacle):
        xmin, ymin, dx_rec, dy_rec, rot_cx, rot_cy, angle = obstacle_obj.get_gmsh_rectangle_params(t_val)
        unrotated_rect = model.occ.addRectangle(xmin, ymin, 0, dx_rec, dy_rec)
        if abs(angle) > 1e-9:
            model.occ.rotate([(2, unrotated_rect)], rot_cx, rot_cy, 0, 0, 0, 1, angle)
        obstacle = unrotated_rect
    else:
        gmsh.finalize()
        raise ValueError(f"Obstacle Type '{type(obstacle_obj)}' not supported.")

    # Boolean: Fluid = Channel - Obstacle
    fluid_shape, _ = model.occ.cut([(2, channel)], [(2, obstacle)])
    model.occ.synchronize()

    # TAGGING of the PHYSICAL GROUPS
    model.addPhysicalGroup(2, [fluid_shape[0][1]], name="Fluid")

    inflow_lines, outflow_lines = [], []
    bottom_lines, top_lines = [], []
    obstacle_lines = []

    tol = 1e-6
    for dim, line_tag in model.getEntities(1):
        com = model.occ.getCenterOfMass(dim, line_tag)
        x_c, y_c = com[0], com[1]

        if np.isclose(x_c, 0.0, atol=tol):
            inflow_lines.append(line_tag)
        elif np.isclose(x_c, Lx, atol=tol):
            outflow_lines.append(line_tag)
        elif np.isclose(y_c, 0.0, atol=tol):
            bottom_lines.append(line_tag)
        elif np.isclose(y_c, Ly, atol=tol):
            top_lines.append(line_tag)
        else:
            obstacle_lines.append(line_tag)

    # Boundary IDs
    if inflow_lines: model.addPhysicalGroup(1, inflow_lines, 1, name="Inflow")
    if outflow_lines: model.addPhysicalGroup(1, outflow_lines, 2, name="Outflow")
    if bottom_lines: model.addPhysicalGroup(1, bottom_lines, 3, name="Bottom")
    if top_lines: model.addPhysicalGroup(1, top_lines, 4, name="Top")
    if obstacle_lines: model.addPhysicalGroup(1, obstacle_lines, 5, name="Obstacle")

    model.mesh.setSize(model.getEntities(0), res_domain)
    
    # obs_points = model.getAdjacencies(1, obstacle_lines[0])[1] if obstacle_lines else []
    for line in obstacle_lines:
        pts = model.getBoundary([(1, line)], combined=False)
        for pt in pts:
            model.mesh.setSize([pt], res_obstacle)

    model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    import os
    tmp_msh_file = f"tmp_mesh_{os.getpid()}_{t_val:.4f}.msh"
    gmsh.write(tmp_msh_file)
    gmsh.finalize()

    m = Mesh(tmp_msh_file)

    if os.path.exists(tmp_msh_file):
        os.remove(tmp_msh_file)

    return m


def create_solid_mesh(obstacle_obj, n):
    """
    Creates a Firedrake mesh for the solid obstacle.
    """
    gmsh.initialize()
    gmsh.model.add("solid_obstacle")
    
    res = 0.01 # Default resolution

    if isinstance(obstacle_obj, circleObstacle):
        x_obs, y_obs, r_obs = obstacle_obj.x_obs, obstacle_obj.y_obs, obstacle_obj.r
        gmsh.model.occ.addDisk(x_obs, y_obs, 0, r_obs, r_obs)
        res = r_obs / n
    elif isinstance(obstacle_obj, squareObstacle):
        x_obs, y_obs, side = obstacle_obj.x_obs, obstacle_obj.y_obs, obstacle_obj.side_length
        half_side = side / 2.0
        gmsh.model.occ.addRectangle(x_obs - half_side, y_obs - half_side, 0, side, side)
        res = side / n
    elif isinstance(obstacle_obj, (lineObstacle, rotatingLineObstacle)):
        # For a line, we create a thin rectangle. This requires the get_gmsh_rectangle_params method.
        try:
            xmin, ymin, dx, dy, rot_cx, rot_cy, angle = obstacle_obj.get_gmsh_rectangle_params(0.0)
            unrotated_rect = gmsh.model.occ.addRectangle(xmin, ymin, 0, dx, dy)
            if abs(angle) > 1e-9:
                gmsh.model.occ.rotate([(2, unrotated_rect)], rot_cx, rot_cy, 0, 0, 0, 1, angle)
            gmsh.model.occ.synchronize() # Synchronize after creating geometry
            res = obstacle_obj.thickness / 2.0
        except AttributeError:
            gmsh.finalize()
            raise TypeError(f"Obstacle type {type(obstacle_obj)} is missing the 'get_gmsh_rectangle_params' method required for solid mesh creation.")
    else:
        gmsh.finalize()
        raise TypeError(f"Obstacle type {type(obstacle_obj)} not supported for solid mesh creation.")

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
    gmsh.model.mesh.generate(2)
    
    # Use a unique temporary file to avoid race conditions
    tmp_solid_msh = f"temp_solid_{os.getpid()}.msh"
    gmsh.write(tmp_solid_msh)
    gmsh.finalize()
    
    try:
        mesh = Mesh(tmp_solid_msh)
        return mesh
    finally:
        if os.path.exists(tmp_solid_msh):
            os.remove(tmp_solid_msh)
		

class create_fluid_mesh:
    def __init__(self, Lx, Ly, n):
        self.Lx = Lx
        self.Ly = Ly
        nnn = n
        ny = int(nnn * (self.Ly / self.Lx))
        self.mesh = RectangleMesh(nnn, ny, self.Lx, self.Ly)
