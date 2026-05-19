import gmsh
import numpy as np
from firedrake import Mesh

def generate_conforming_mesh(Lx, Ly, x_obs, y_obs, r_obs, n):
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
