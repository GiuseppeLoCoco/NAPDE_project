from firedrake import File, CheckpointFile
from os import path, makedirs, listdir, remove
from shutil import rmtree
import io

class write_mesh:
    def __init__(self, directory, mesh, filename):
        folder = path.join(directory, "mesh_files/")
        makedirs(folder, exist_ok=True)
        # Firedrake usa .pvd (VTK) come standard di visualizzazione
        self.file = File(folder + filename + ".pvd")
        self.file.write(mesh)
        
    def write_mesh_boundaries(self, boundaries):
        pass # In Firedrake sono integrati nella mesh o si scrivono le marker functions

    def write_mesh_subdomains(self, subdomains):
        pass

class write_mesh_H5:
    def __init__(self, mpi_comm, directory, mesh, filename):
        folder = path.join(directory, "restart_variables/")
        makedirs(folder, exist_ok=True)
        # Si tiene un riferimento aperto al file di checkpoint
        self.hdf = CheckpointFile(folder + filename + ".h5", "w", comm=mpi_comm)
        self.hdf.save_mesh(mesh)
        self.hdf.close()
        
    def write_mesh_H5_boundaries(self, boundaries):
        pass		

    def write_mesh_H5_subdomains(self, subdomains):
        pass

def xdmf_file(directory, filename, rewrite_mesh):
    # Ritorna l'handler per i file PVD di Firedrake (sostituto di XDMFFile)
    return File(directory + filename + ".pvd")

class create_result_folder:
    def __init__(self, directory, restart, dim, bool_test, suffix=""):
        folder = path.join(directory, "results" + suffix + "/")
        if not restart:
            try:
                makedirs(folder, exist_ok=True)
            except OSError:
                pass 

        self.bool_stream = (dim == 2 and bool_test)
        self.folder = folder
        self.restart = restart

    def create_files(self, files, mpi_comm):
        xdmf_file_handles = dict()
        folder_xdmf = path.join(self.folder, "PVD_files/")
        makedirs(folder_xdmf, exist_ok=True)

        for i in files:
            xdmf_file_handles[i] = File(folder_xdmf + i + ".pvd")

        # In Firedrake non pre-allocchiamo i Checkpoint file qui, 
        # li apriamo/chiudiamo durante la funzione di salvataggio per evitare corruzioni.
        hdf5_file_handles = files # Passiamo semplicemente le chiavi

        return xdmf_file_handles, hdf5_file_handles	

    def create_text_files(self, text_files, my_rank):	
        if not self.restart:
            makedirs(self.folder + "text_files/", exist_ok=True)

        text_file_handles = []	
        for i in text_files:
            handle = io.TextIOWrapper(open(self.folder + "text_files/" + i + ".txt", "ab+", 0), write_through=True)
            text_file_handles.append(handle)

        if not self.restart and my_rank == 0:
            for i in text_file_handles:
                i.truncate(0)
                i.seek(0)			

        return text_file_handles

    def write_header_text_files(self, text_file_handles, my_rank):
        if not self.restart and my_rank == 0:
            text_file_handles[0].write("#Time		#Drag			#Lift\n")
            text_file_handles[1].write("#Time             #Timestep         #PISO velocity_error     #Max cell_Re         #Max Convection_CFL  #Max Viscous_CFL     #Max Conduction_CFL\n")
            try:
                text_file_handles[4].write("#Time		#Average_nusselt_no.\n")
                text_file_handles[5].write("#Time		#Drag			#Lift			#Volume\n")
                text_file_handles[6].write("#Time		#Drag			#Lift\n")
                text_file_handles[7].write("#Time		#Radius_ratio min	#Radius_ratio max\n")
                text_file_handles[8].write("#Time		#Average_nusselt_no.\n")
            except:
                pass	

def write_restart_files(directory, Mpi, file_handle, t, tsp, **restart_variables):
    if Mpi.get_rank() == 0:
        file_handle.truncate(0)
        file_handle.seek(0)
        file_handle.write(f"{t}\n{tsp}")

    for key, value in restart_variables.items():
        y = 0
        for i in value:
            y += 1
            with CheckpointFile(directory + f"restart_variables/{key}_variable_{y}.h5", "w", comm=Mpi.mpi_comm) as hdf:
                hdf.save_function(i, name="variable")

def write_solution_files(problem_physics, bool_stream, t, xdmf_file_handles, hdf5_file_handles, **variables):
    flow_variables = variables.get('flow', {})
    solid_variables = variables.get('solid', {})
    lagrange_variables = variables.get('lagrange', {})
    
    u = flow_variables.get('uv')
    p = flow_variables.get('p_')[0] if 'p_' in flow_variables else None
    
    if u is not None:
        u.rename("velocity")
        xdmf_file_handles['u'].write(u, time=t)
    if p is not None:
        p.rename("pressure")
        xdmf_file_handles['p'].write(p, time=t)

    if bool_stream:
        vort = flow_variables.get('vort')
        psi = flow_variables.get('psi')
        vort.rename("vorticity")
        xdmf_file_handles['vorticity'].write(vort, time=t)
        psi.rename("stream_function")
        xdmf_file_handles['stream_function'].write(psi, time=t)

    if problem_physics['solve_FSI']:
        Dp = solid_variables.get('Dp_')[0]
        us = solid_variables.get('us_')
        Lm = lagrange_variables.get('Lm_')[0]
        
        Dp.rename("displacement")
        xdmf_file_handles['Dp'].write(Dp, time=t)
        us.rename("velocity_solid")
        xdmf_file_handles['us'].write(us, time=t)
        Lm.rename("lagrange-multiplier")
        xdmf_file_handles['Lm'].write(Lm, time=t)