from firedrake import Mesh, Function, VectorFunctionSpace, FunctionSpace, File, SpatialCoordinate, CheckpointFile
import numpy as np
import os, csv, operator
from os import listdir, path, makedirs
import gmsh

class get_mesh:
    def __init__(self, mpi_comm, directory, filename):
        # Firedrake usa CheckpointFile per leggere file h5 salvati precedentemente
        with CheckpointFile(directory + filename, "r", comm=mpi_comm) as hdf:
            self.mesh = hdf.load_mesh("mesh")

    def get_mesh_boundaries(self):
        # In Firedrake, i marker sono caricati nativamente con la mesh dal CheckpointFile
        # Ritorna None poichè l'integrazione ds(marker) li preleva in automatico dalla mesh
        return None

    def get_mesh_subdomains(self):
        return None

def read_boundary_conditions(directory, file_name):
    filename = directory + file_name
    file = csv.reader(open(filename, 'r'))
    xvalues = []
    yvalues = []
    xmin = 10000.0
    xmax = -10000.0
    for row in file:
        xval = float(row[0])
        yval = float(row[1])
        if xval < 0: xval = 0
        if xval < xmin: xmin = xval
        if xval > xmax: xmax = xval
        xvalues.append(xval)
        yvalues.append(yval)
        
    size = len(xvalues)
    xdmf = xvalues[-1] - xvalues[0]
    xvalues.append(xvalues[0] + xdmf / (size - 1) * size)
    yvalues.append(yvalues[0])
    
    if np.any(np.diff(xvalues) < 0):
        L = sorted(zip(xvalues, yvalues), key=operator.itemgetter(0))
        xvalues, yvalues = zip(*L)

    return np.array(xvalues), np.array(yvalues)		

def read_restart_files(directory, mpi_comm, file_handle, **restart_variables):
    for key, value in restart_variables.items():
        y = 0
        for i in value:
            y += 1
            with CheckpointFile(directory + f"restart_variables/{key}_variable_{y}.h5", "r", comm=mpi_comm) as hdf:
                # Firedrake carica direttamente la funzione nella variabile esistente
                i.assign(hdf.load_function(i.function_space(), name="variable"))

    t, tsp = 0.0, 0.0
    file_handle.seek(0)
    t = float(file_handle.readline())
    tsp = float(file_handle.readline())  
    return t, tsp

class get_mesh_gmsh:
    def __init__(self, mpi_comm, directory, basename):
        # Firedrake legge direttamente i file .msh di Gmsh importando le fisiche/markers
        self.mesh = Mesh(directory + basename + ".msh", comm=mpi_comm)
		
    def get_mesh_boundaries(self):
        return None

    def get_mesh_subdomains(self):
        return None

class create_fluid_mesh:
    def __init__(self):
        self.Lx, self.Ly = 2.2, 0.41
        nnn = 70
        # In Firedrake RectangleMesh è nativo e non serve mshr
        ny = int(nnn * (self.Ly / self.Lx))
        self.mesh = RectangleMesh(nnn, ny, self.Lx, self.Ly)

    def get_mesh_boundaries(self):
        # In Firedrake, il mesh builder rettangolare imposta nativamente i markers:
        # 1: x=0 (Inflow), 2: x=Lx (Outflow), 3: y=0, 4: y=Ly (Walls)
        # Basterà usare ds(1), ds(2), ds((3,4)) nelle variazionali
        return None

class create_solid_mesh:
    def __init__(self):
        x_obs, y_obs, r_obs = 0.2, 0.2, 0.05
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
		
    def get_mesh_boundaries(self):
        return None