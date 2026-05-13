from dolfin import XDMFFile, HDF5File, TimeSeries
from os import path, makedirs, listdir, remove
from shutil import rmtree
import io

def poisson_write_solution_files(problem_physics, bool_stream, k, xdmf_file_handles, hdf5_file_handles, **variables):

	for key, value in variables.items():
		if key == 'poisson':
			poisson_variables = variables['poisson']
		if key == 'lagrange':
			lagrange_variables = variables['lagrange']

	# --------------------------------			

	u = poisson_variables['u_']
	Lm_f = poisson_variables['Lm_']

	Lm = lagrange_variables['Lm_']
	uf_ = lagrange_variables['uf_']

	# --------------------------------		

	# Save solution to file (HDF5_)

	hdf5_file_handles['u'].write(u, 'u', k); hdf5_file_handles['u'].flush()
	
	# hdf5_file_handles['us'].write(us, 'us', k); hdf5_file_handles['us'].flush()
	hdf5_file_handles['Lm'].write(Lm, 'lagrange-multiplier', k); hdf5_file_handles['Lm'].flush()
		
	# --------------------------------		    	    

	# Save solution to file (XDMF)

	u.rename('u', 'u')
	xdmf_file_handles['u'].write(u, k)
	Lm_f.rename('Lm_f', 'Lm_f')
	xdmf_file_handles['u'].write(Lm_f, k)
	xdmf_file_handles['u'].close()

	# us.rename('us', 'us')
	# xdmf_file_handles['us'].write(us, 0.0); xdmf_file_handles['us'].close()
	Lm.rename('lagr', 'lagr')
	xdmf_file_handles['Lm'].write(Lm, k)
	uf_.rename('uf_', 'uf_')
	xdmf_file_handles['Lm'].write(uf_, k)
	xdmf_file_handles['Lm'].close()