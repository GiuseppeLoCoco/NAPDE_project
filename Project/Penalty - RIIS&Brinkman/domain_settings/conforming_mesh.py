import numpy as np
from firedrake import Mesh

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

    # Added part (see better)

    # 1. Forza GMSH a usare il formato MSH versione 2 (altamente raccomandato per Firedrake)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    # 2. Salva la mesh in un file temporaneo
    mesh_filename = "mesh_conforming_temp.msh"
    gmsh.write(mesh_filename)
    
    # 3. Finalizza GMSH per liberare la memoria ed evitare conflitti in run successivi
    gmsh.finalize()

    # 4. Inizializza la Mesh di Firedrake passandogli il PATH del file appena creato
    m = Mesh(mesh_filename)

    import os
    if os.path.exists(mesh_filename):
        os.remove(mesh_filename)
    
    return m
