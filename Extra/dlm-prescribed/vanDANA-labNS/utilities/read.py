from dolfin import Mesh, HDF5File, MeshFunction, File, TimeSeries, \
			VectorFunctionSpace, FunctionSpace, XDMFFile, Function, FunctionAssigner 
import numpy as np
from os import listdir, path, makedirs
import csv

class get_mesh:

	def __init__(self, mpi_comm, directory, filename):

		mesh = Mesh(mpi_comm)		
		hdf = HDF5File(mpi_comm, directory + filename, "r")		
		hdf.read(mesh, "/mesh", False)
		
		self.hdf = hdf
		self.mesh = mesh
		
	def get_mesh_boundaries(self):

		boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
		self.hdf.read(boundaries, "/boundaries")		
		return boundaries

	def get_mesh_subdomains(self):

		subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
		self.hdf.read(subdomains, "/subdomains")
		return subdomains




def read_boundary_conditions(directory, file_name):

	filename = directory + file_name
	file=csv.reader(open(filename,'r'))
	xvalues=[]
	yvalues=[]
	xmin = 10000.0
	xmax = -10000.0
	for row in file:
	    xval = float(row[0])
	    yval = float(row[1])
	    if(xval<0):
	        xval=0
	    if (xval<xmin):
	        xmin = xval
	    if (xval>xmax):
	        xmax = xval
	    xvalues.append(xval)
	    yvalues.append(yval)
	size = len(xvalues)
	xdiff = xvalues[size-1]-xvalues[0]
	xvalues.append(xvalues[0]+xdiff/(size-1)*size)
	yvalues.append(yvalues[0])
	if np.any(np.diff(xvalues) < 0):
	    L = sorted(zip(xvalues,yvalues), key=operator.itemgetter(0))
	    xvalues, yvalues = zip(*L)

	return np.array(xvalues), np.array(yvalues)		



def read_time_series(mpi_comm, directory):

	mesh = Mesh(mpi_comm)

	folder = path.join(directory, "results/mesh_files/")
	try:
	    makedirs(path.join(folder, 'time_series_solid_current_mesh/'), exist_ok = True)
	except OSError:
	    pass

	pvd_file = File(path.join(folder, 'time_series_solid_current_mesh/') + 'solid_current_mesh.pvd')    

	for k in range(0, 250):
		nm = 'time_series_solid_current_mesh_' + str(k) + '.h5'
		timeseries = TimeSeries(folder + 'time_series_solid_current_mesh_' + str(k))		
		
		if nm in listdir(folder):	
			times = timeseries.mesh_times()
			for tt in times:
				timeseries.retrieve(mesh, tt)
				pvd_file << (mesh, tt)
		else:
			break



def read_restart_files(directory, mpi_comm, file_handle, **restart_variables):
		
	for key, value in restart_variables.items():
		y = 0
		for i in value:
			y += 1
			hdf = HDF5File(mpi_comm, directory + "restart_variables/" + str(key) + "_variable_" + str(y) + ".h5", "r")
			hdf.read(i, "/variable/vector_0"); del hdf

	t = 0; tsp = 0
	file_handle.seek(0)
	t = float(file_handle.readline())
	tsp = float(file_handle.readline())  

	return t, tsp



def extract_hdf5_data_for_xdmf_visualization(mpi_comm, curr_dir, bool_stream, problem_physics, fem_degree):

	extract_hdf5_to_xdmf(mpi_comm, curr_dir, "u", "file_f.h5", fem_degree['velocity_degree'], "velocity", "v", False)
	extract_hdf5_to_xdmf(mpi_comm, curr_dir, "p", "file_f.h5", fem_degree['pressure_degree'], "pressure", "s", False)
	
	if problem_physics['solve_temperature'] == True:
		extract_hdf5_to_xdmf(mpi_comm, curr_dir, "T", "file_f.h5", fem_degree['temperature_degree'], "temperature", "s", False)

	if bool_stream == True:
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "vorticity", "file_f.h5", fem_degree['pressure_degree'], "vorticity", "s", False)
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "stream_function", "file_f.h5", fem_degree['pressure_degree'], "stream_function", "s", False)

	if problem_physics['solve_FSI'] == True:
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "Dp", "file_s.h5", fem_degree['displacement_degree'], "displacement", "v", False)
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "us", "file_s.h5", fem_degree['displacement_degree'], "velocity_solid", "v", False)
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "ps", "file_s.h5", fem_degree['pressure_degree'], "pressure_solid", "s", False)
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "J", "file_s.h5", fem_degree['pressure_degree'], "Jacobian", "s", False)
	    extract_hdf5_to_xdmf(mpi_comm, curr_dir, "Lm", "file_s.h5", fem_degree['lagrange_degree'], "lagrange-multiplier", "v", False)
	    
	    if problem_physics['solve_temperature'] == True:
	    	extract_hdf5_to_xdmf(mpi_comm, curr_dir, "Ts", "file_s.h5", fem_degree['temperature_degree'], "temperature_solid", "s", False)



def extract_hdf5_to_xdmf(mpi_comm, directory, filename, meshfile, deg_FS, fieldvariable, c, rewrite_mesh):

	mesh = Mesh(mpi_comm)
	hdf_mesh = HDF5File(mpi_comm, directory + "user_inputs/" + meshfile , "r")
	hdf_mesh.read(mesh,"/mesh",False)
	del hdf_mesh

	# --------------------------------------------------------------------------------

	if c == "s":
	    space = FunctionSpace(mesh, "CG", deg_FS)
	elif c == "v":
	    space = VectorFunctionSpace(mesh, "CG", deg_FS)

	var = Function(space)

	# --------------------------------------------------------------------------------

	file = XDMFFile(directory + "results/XDMF_files/" + filename + ".xdmf")
	file.parameters['flush_output'] = True; file.parameters['rewrite_function_mesh'] = rewrite_mesh

	for k in range(250):

		nm = 'HDF5_files_' + str(k)
		if nm in listdir(directory + "results/"):

			hdf = HDF5File(mpi_comm, directory + "results/" + nm + "/" + filename + "_.h5", "r")
			if hdf.has_dataset(fieldvariable):
				attr = hdf.attributes(fieldvariable)
				nsteps = attr['count']

				for i in range(nsteps):

					dataset = fieldvariable+"/vector_%d"%i
					attr = hdf.attributes(dataset)
					t = attr['timestamp']
					hdf.read(var, dataset)

					var.rename(fieldvariable, fieldvariable)
					file.write(var, t)

				hdf.close(); del hdf

		else:
			break

	file.close()    



class get_mesh_gmsh:

	def __init__(self, mpi_comm, directory, basename):

		mesh = Mesh(mpi_comm, directory + basename + ".xml")
		boundaries = MeshFunction("size_t", mesh, directory + basename + "_facet_region.xml")
		subdomains = MeshFunction("size_t", mesh, directory + basename + "_physical_region.xml")

		self.mesh       = mesh
		self.boundaries = boundaries
		self.subdomains = subdomains
		
	def get_mesh_boundaries(self):

		return self.boundaries

	def get_mesh_subdomains(self):

		return self.subdomains

from mshr import *
from dolfin import *
class create_fluid_mesh:

	def __init__(self):

		self.Lx, self.Ly = 2.2,0.41#   3, 1#11, 4.1 #3, 1
		domain = Rectangle(Point(0, 0), Point(self.Lx, self.Ly))
		nnn = 70
		mesh = generate_mesh(domain, nnn)
		
		self.hdf = None
		self.mesh = mesh

	class Inflow(SubDomain):
		def __init__(self):
			SubDomain.__init__(self)
		def inside(self, x, on_boundary):
			return on_boundary and near(x[0], 0)
	class Outflow(SubDomain):
		def __init__(self, Lx):
			SubDomain.__init__(self)
			self.Lx = Lx
		def inside(self, x, on_boundary):
			return on_boundary and near(x[0], self.Lx)
	class Walls(SubDomain):
		def __init__(self, Ly):
			SubDomain.__init__(self)
			self.Ly = Ly
		def inside(self, x, on_boundary):
			return on_boundary and (near(x[1], 0) or near(x[1], self.Ly))

	def get_mesh_boundaries(self):

		boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
		self.Inflow().mark(boundaries, 1)
		self.Outflow(self.Lx).mark(boundaries, 2)
		self.Walls(self.Ly).mark(boundaries, 3)
		return boundaries

	def get_mesh_subdomains(self):

		subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
		return subdomains

class create_solid_mesh:

	def __init__(self):

		x_obs = 0.2 #0.5#2#0.5
		y_obs = 0.2 #0.5#2#0.5
		r_obs = 0.05 #0.1#0.5#0.1
		domain = Circle(Point(x_obs, y_obs), r_obs)
		nnn = int(70*3*r_obs/2.2)	# same h of the fluid mesh, when fluid mesh is [0,2.2]x[0,0.41]
		mesh = generate_mesh(domain, nnn)
		
		self.hdf = None
		self.mesh = mesh
		
	def get_mesh_boundaries(self):

		boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
		return boundaries

	def get_mesh_subdomains(self):

		subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
		return subdomains
