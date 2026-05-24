from firedrake import Mesh, Function, VectorFunctionSpace, FunctionSpace, File, SpatialCoordinate, CheckpointFile
import numpy as np
import os, csv, operator
from os import listdir, path, makedirs
import gmsh

def conforming_mesh(Lx, Ly, x_obs, y_obs, r_obs, n):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Cylinder")

    # Resolution
    res = Ly / n

    # Geometry
    channel = model.occ.addRectangle(0, 0, 0, Lx, Ly)
    cylinder = model.occ.addDisk(x_obs, y_obs, 0, r_obs, r_obs)

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
        if np.isclose(mass_center[0], 0): model.addPhysicalGroup(1, [line[1]], 3)    # Inflow
        elif np.isclose(mass_center[0], Lx): model.addPhysicalGroup(1, [line[1]], 4) # Outflow
        elif np.isclose(mass_center[1], 0): model.addPhysicalGroup(1, [line[1]], 1)  # Bottom
        elif np.isclose(mass_center[1], Ly): model.addPhysicalGroup(1, [line[1]], 2) # Top
        else: model.addPhysicalGroup(1, [line[1]], 5)                                # Cylinder Boundary

    model.mesh.setSize(model.getEntities(0), res)
    model.mesh.generate(2)

    m = Mesh(model)
    gmsh.finalize()
    return m


class create_solid_mesh:
    def __init__(self,x_obs, y_obs, r_obs):
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
    def __init__(self,Lx,Ly,n):
        self.Lx = Lx
        self.Ly = Ly
        nnn = n
        ny = int(nnn * (self.Ly / self.Lx))
        self.mesh = RectangleMesh(nnn, ny, self.Lx, self.Ly)
