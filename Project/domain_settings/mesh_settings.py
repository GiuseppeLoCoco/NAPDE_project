from firedrake import Mesh, RectangleMesh
import numpy as np
import os, csv, operator
from os import listdir, path, makedirs
import gmsh # Keep gmsh import here for the Mesh(model) constructor and other gmsh calls

from .obstacles import circleObstacle, squareObstacle, rotatingLineObstacle # Import obstacle types

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


class create_solid_mesh:
    def __init__(self, x_obs, y_obs, r_obs):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs
        nnn = int(70 * 3 * r_obs / 2.2)
        
        # Per geometrie circolari/curve, in Firedrake lo standard è Gmsh API
        gmsh.initialize()
        gmsh.model.add("circle")
        gmsh.model.occ.addDisk(x_obs, y_obs, 0, r_obs, r_obs)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", r_obs/nnn)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", r_obs/nnn)
        gmsh.model.mesh.generate(2)
        
        gmsh.write("temp_circle.msh")
        gmsh.finalize()
        
        self.mesh = Mesh("temp_circle.msh")
        if os.path.exists("temp_circle.msh"):
            os.remove("temp_circle.msh")
		

class create_fluid_mesh:
    def __init__(self, Lx, Ly, n):
        self.Lx = Lx
        self.Ly = Ly
        nnn = n
        ny = int(nnn * (self.Ly / self.Lx))
        self.mesh = RectangleMesh(nnn, ny, self.Lx, self.Ly)
