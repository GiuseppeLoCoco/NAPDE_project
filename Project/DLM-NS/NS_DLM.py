from firedrake import *
from utilities import *
from user_inputs import *
from common import *
from time import time
import vtk_py3 as vtk_py3
import numpy as np
import array as arr
import math, os, operator, copy, sys, io, json, vtk, matplotlib, cppimport, argparse, traceback
matplotlib.use('Agg')
from matplotlib import rc, pylab as plt
import inspect
from ast import Constant
from symtable import Function

def saveVTK(file_dict, t, uh, ph, phi, chi):
    uh.rename('u','u')
    ph.rename('p','p')
    phi.rename('phi','phi')
    chi.rename('chi','chi')
    file_dict['u'].write(uh, time=t)
    file_dict['p'].write(ph, time=t)
    file_dict['phi'].write(phi, time=t)
    file_dict['chi'].write(chi, time=t)

class NS_DLM_Solver:

    def __init__(self, conforming=False, moving=True):
        self.conforming = conforming
        self.moving = moving

    def NS_DLM_Solve(self, args=None):

        timer_total.start()

        # --------------------------------

        fem_degree.update({"velocity_degree": args.velocity_degree})

        keep_solid_on_reference = False
        corrective_step = False

        curr_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        remove_killvanDANA(curr_dir)
        remove_complete(curr_dir)

        blockPrint()
                        
        # --------------------------------
        # Create meshes for fluid and solid (for DLM) domains
        # --------------------------------

        Lx, Ly = 3, 1
        y_obs = 0.5
        r_obs = 0.1
        n = 50

        # Create the mesh
        if self.conforming:
            mesh = conforming_mesh(Lx, Ly, y_obs, y_obs, r_obs, n)
        else:
            fluid_mesh = create_fluid_mesh()
            solid_mesh = create_solid_mesh()

        hmax_f = Max(fluid_mesh.mesh.hmax())
        hmin_f = Min(fluid_mesh.mesh.hmin())
        hmax_s = Max(solid_mesh.mesh.hmax()) 
        hmin_s = Min(solid_mesh.mesh.hmin())

        # --------------------------------

        print("\nFluid mesh specs | edge length: Max =",hmax_f, "; Min =",hmin_f, flush = True)
        print("\nSolid mesh specs | edge length: Max =",hmax_s, "; Min =",hmin_s, flush = True)

        # --------------------------------

        # ================================

        # Calculate non-dimensional numbers of the fluid problem
        allowed_keys = inspect.signature(calc_non_dimensional_numbers).parameters.keys()                  # Get the parameter names of the function
        filt_physical_parameters = {k: v for k, v in physical_parameters.items() if k in allowed_keys}    # Filter dicts to only include keys in the function's parameters
        Re, _, _, Fr = calc_non_dimensional_numbers(**filt_physical_parameters, **characteristic_scales)
        # See better this part (may need some not-yet-defined functions)
        print(f"Re={Re}, Fr={Fr}")

        # --------------------------------

        # Time step setting
        tsp = dt = time_control['dt']
        T = time_control['T']
        dt = Constant(dt)
        # See better this part (may need some not-yet-defined functions)

        print(RED % "\nInitial time_step = {}".format(tsp), flush = True)

        # ================================

        # --------------------------------
        # Initialize Flow Variational Problem
        # --------------------------------

        # Problem dimension
        dim = fluid_mesh.mesh.geometric_dimension()
        u_components = dim
        self.u_components = u_components
        # Define function spaces
        V = FunctionSpace(mesh, 'P', fem_degree['velocity_degree'])		  
        Q = FunctionSpace(mesh, 'P', fem_degree['pressure_degree'])		  
        Z1 = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                            

        Vp = VectorFunctionSpace(mesh, 'P', fem_degree['velocity_degree'])

        u1 = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        Lm1 = [TrialFunction(Z1.sub(ui)) for ui in range(u_components)]

        self.u1 = u1
        self.v = v
        self.p = p
        self.q = q
        self.Lm1 = Lm1

        variables = dict()
        u_ = []
        p_ = []
        for i in range(3):
            u_.insert(i, as_vector([Function(V) for ui in range(u_components)]))
            p_.insert(i, Function(Q))

        uv = Function(Vp)		
        Lm_f = Function(Z1)
        Lm_f.assign(0.0)

        f = Constant(tuple([0.0]*dim))

        variables.update(u_=u_, uv=uv, p_=p_, Lm_f=Lm_f)	

        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.matrix = dict(Mij=None, Kij=None, Cij=None,
                           Sij=[None for ui in range(u_components)],
                           Bij=[None for ui in range(u_components)],
                           Pij=[None for ui in range(u_components)],
                           Yij=[None for ui in range(u_components)],
                           A1_as1=None, A2_as2=None, b1_Ls1=None, A1_SCW1=None, b2=None)
        
        u_ab = as_vector([Function(V) for ui in range(self.u_components)]) 

        self.u_ab = u_ab
        self.f = f
        self.dim = dim
        self.variables = variables

        u_solver_params = {
            "ksp_type": tentative_velocity_solver['solver_type'],
            "pc_type": tentative_velocity_solver['preconditioner_type']
        }

        p_solver_params = {
            "ksp_type": pressure_correction_solver['solver_type'],
            "pc_type": pressure_correction_solver['preconditioner_type']
        }

        u_c_solver_params = {
            "ksp_type": velocity_correction_solver['solver_type'],
            "pc_type": velocity_correction_solver['preconditioner_type']
        }

        F = [V, Q, Z1]

        FS = dict(fluid = F)

        # Ausiliary variable for the previous Lagrange multiplier
        # (used for velocity correction and then update in the corrective step)
        Lm_f_old = Function(Lm_f.function_space())

        # --------------------------------
        # Prescribed kinematics for the solid
        # --------------------------------

        mesh = solid_mesh.mesh
        dim = mesh.geometric_dimension()

        R = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])  
        Z = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])      

        test = TestFunction(R) 
        solid_variables = dict()

        us_ = Function(R)
        solid_variables.update(us_=us_)

        F = [R, Z]
        n = FacetNormal(mesh)
        nx = tensors.unit_vector(0, dim)
        ny = tensors.unit_vector(1, dim)
        dx = Measure("dx", domain=mesh)
        
        # Compute Amplitude
        coords = mesh.coordinates.dat.vec_ro.array
        self.diameter = np.linalg.norm(coords.max() - coords.min())
        self.amplitude = 0.0 * self.diameter

        # --------------------------------
        # Initialize Lagrage Multiplier Variational Problem
        # --------------------------------

        mesh = solid_mesh.mesh
        dim = mesh.geometric_dimension()

        Y = VectorFunctionSpace(mesh, 'P', fem_degree['displacement_degree'])                                             
        Z2 = VectorFunctionSpace(mesh, 'P', fem_degree['lagrange_degree'])                                                 

        self.Lm = TrialFunction(Z2)
        self.e = TestFunction(Z2)

        lagrange_variables = dict()
        uf_ = Function(Y)
        us_ = Function(Y)
        us_.assign(0.0)
		
        Lm_ = [Function(Z2), Function(Z2)]
        Lm_[0].assign(0.0); Lm_[1].assign(0.0)
        
        lagrange_variables.update(Lm_=Lm_, uf_=uf_)	

        F = [Y, Z2]
        nx = tensors.unit_vector(0, dim)
        ny = tensors.unit_vector(1, dim)
        dx = Measure("dx", domain=mesh)

        FS.update(lagrange = F)

        variables = dict(variables)
        variables.update(solid=solid_variables, lagrange=lagrange_variables)

        self.us_ = us_
        self.F = [Y, Z2]
        self.nx = tensors.unit_vector(0, dim)
        self.ny = tensors.unit_vector(1, dim)
        self.dx = Measure("dx", domain=mesh)

        # ================================

        # Create boundary conditions for the fluid problem
        bcs = prescrKin_fluid_create_boundary_conditions(fluid_mesh, **FS)

        # ---------------------------------

        # Delta-interpolation for the Fluid-Structure interaction 
        fsi_interpolation = compile_cpp_code(fsi_interpolation_code)
        fsi_interpolation.create_bounding_box(solid_mesh.mesh)
        fsi_interpolation.calculate_fluid_mesh_size_h(fluid_mesh.mesh)
        fsi_interpolation.extract_dof_component_map_user(FS['fluid'][2], "F")
        fsi_interpolation.extract_dof_component_map_user(FS['lagrange'][0], "S")

        # ---------------------------------

        # Initialize the time
        t = 0.0

        # Enter number of counters required
        counters = create_counters(5) 
        s1, s2, s3, s4, si, sr, s_dt = [0.0 for _ in range(10)]

        timeseries = write_time_series(result_folder.folder, restart)

        """
        # Output/write meshes
        pv1 = write_mesh(result_folder.folder, fluid_mesh.mesh, "fluid_mesh")
        pv1.write_mesh_boundaries(fluid_mesh.get_mesh_boundaries())

        timeseries = write_time_series(result_folder.folder, restart)

        filename = "solid_reference_mesh"
        if restart == True:	filename = "solid_restart_mesh"

        pv = write_mesh(result_folder.folder, solid_mesh.mesh, filename)
        """ 

        # Output files for the solutions
        files = ['u', 'p']
        text_files = ['flow_data', 'runtime_stats', 'restart', 'log_info']
        files.extend(['us', 'Lm'])
        text_files.extend(['solid_data', 'lagrange_data'])

        if result_folder.bool_stream == True:
            files.extend(['vorticity', 'stream_function'])

        xdmf_file_handles, hdf5_file_handles = result_folder.create_files(files)
        text_file_handles = result_folder.create_text_files(text_files)
        result_folder.write_header_text_files(text_file_handles)

        write_solution_files(problem_physics, result_folder.bool_stream, t, xdmf_file_handles, hdf5_file_handles, **variables)

        print(RED % "Total time = {}".format(T), "\n", flush = True)

        # ---------------------------------

        # Calculate total Degrees of Freedom
        DOFS_variables = dict(velocity = [u_[0][ui] for ui in range(u_components)], pressure = [p_[0]])
        DOFS_variables.update(lagrange_multiplier = [Lm_[0]])

        DOFS = Calc_total_DOF(**DOFS_variables)
        print(GREEN % 'DOFs = {}'.format(DOFS), "\n", flush = True)

        # ---------------------------------

        # Update temporal variables
        update = [u_, p_]
        update.extend([Lm_])

        print(RED % 'Start Simulatons : t = {}'.format(t), "\n", flush = True)

        # ================================

        # ----------------------------------
        # Time-stepping Loop
        # ----------------------------------

        try:

            while t < T:

                try: 	

                    # Start the timer to check the runtime of each time step
                    timer_dt.start()

                    # Update time step
                    update_counter(counters)
                    t += tsp

                    # Create the counding box of the solid mesh for the interpolation
                    fsi_interpolation.create_bounding_box(solid_mesh.mesh)

                    # Update the boundary conditions
                    time_varying_bc(t)

                    # Update the Lagrange multiplier for the new time step
                    # Lm_[0] = lambda(t+1) and Lm_[1] = lambda(t)
                    Lm_[1].assign(Lm_[0])

                    # Interpolate the Lagrange multiplier from the solid mesh to
                    # the fluid mesh (for the computation of the tentative velocity)
                    Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))

                    # ----------------------------------

                    print(BLUE % "1: Predict tentative velocity step", flush = True)
                    # Assemble and solve the problem of the tentative velocity step
                    timer_s1.start()
                    A1, b1 = self.assemble_tentative_velocity(u_, p_, Lm_f, dt)
                    self.solve_tentative_velocity(A1, u_[0], b1, bcs['velocity'])
                    s1 += timer_s1.stop()

                    # ================================

                    print(BLUE % "2: Pressure correction step", flush = True)
                    # Assemble and solve the problem of the pressure correction
                    timer_s2.start()
                    A2, b2 = self.assemble_pressure_correction(u_, Lm_f, dt)
                    self.solve_pressure_correction(A2, p_[0], b2, bcs['pressure'])
                    s2 += timer_s2.stop()

                    # ================================

                    # Create the velocity vector
                    assigner_uv.assign(uv, [u_[0][ui] for ui in range(u_components)])

                    # Interpolate the velocity on the solid mesh in order to obtain 
                    # the new Lagrange multiplier lambda(n+1)
                    timer_si.start()
                    uf_.assign(interpolate_nonmetching_mesh_delta(fsi_interpolation, uv, "S"))
                    si += timer_si.stop()

                    # Update the solid position 
                    self.update_solid(solid_mesh.mesh, t, dt)

                    print(BLUE % "3: Lagrange multiplier (fictitious force) step", flush = True)
                    # Assemble and solve the problem of the Lagrange multiplier step
                    timer_s3.start()
                    a3, b3 = self.assemble_lagrange_multiplier(Lm_, us_, uf_, dt)
                    self.solve_lagrange_multiplier(a3, Lm_[0], b3)
                    s3 += timer_s3.stop()

                    # ================================

                    # The final corrective step for velocity
                    Lm_f.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[0], "F"))
                    Lm_f_old.assign(interpolate_nonmatching_mesh_delta(fsi_interpolation, Lm_[1], "F"))
                    bDLM = self.assemble_velocity_correction_DLM(u_[0], Lm_f, Lm_f_old, dt)
                    print(BLUE % "4: Velocity correction step", flush = True)
                    timer_s4.start()
                    self.solve_velocity_correction(u_[0], bDLM, bcs['velocity'])
                    s4 += timer_s4.stop()

                    # ================================

                except Exception as e:

                    print(BLUE % 'error message : ', flush = True); traceback.print_exc(file=sys.stdout) #; print(e, flush = True)
                    print(BLUE % "vanDANA solver diverged --- at time : {} sec , corresponding timestep : {}".format(t, tsp), flush = True)

                    print(BLUE % '\nvanDANA solver - TERMINATED : t = {}'.format(t), "\n", flush = True)

                else:

                    # Update progress on terminal
                    print("Time : t =", round_decimals_down(t, 5), '\t'*12 + "Progress : " + str(round_decimals_down((t/T)*100, 5)) + " %", flush = True)

                    # Print output files
                    if counters[0] >= print_control['a']:

                        reset_counter(counters, 0);
                        print(BLUE % "File printing in progress --- Simulation run time : {} , Wall time elapsed : {} sec".format(t, timer_total.elapsed()[0]), flush = True)

                        vort, psi = self.calc_vorticity_streamfunction(uv, bcs['streamfunction'])

                        write_solution_files(problem_physics, result_folder.bool_stream, t, xdmf_file_handles, hdf5_file_handles, **variables)

                        pv2 = write_mesh_H5(result_folder.folder, solid_mesh.mesh, "solid_current_mesh")
                        pv2.hdf.close()

                        timeseries.store(solid_mesh.mesh, t)

                    # ----------------------------------

                    # If required: calculate new time-step
                    if counters[4] >= print_control['e']:

                        reset_counter(counters, 4)
                        tsp = calc_runtime_stats_timestep(problem_physics, u_[0], u_components, 
                                                          u_diff, t, tsp, text_file_handles, fluid_mesh.mesh, 
                                                          hmin_f, flow.h_f_X, Re, np.nan, np.nan, flow.VN_local, time_control)
                        dt  = Constant(tsp)

                    # ----------------------------------

                    # Update previous solution
                    update_variables(update, u_components, problem_physics)

                finally:

                    # Timing tasks
                    if counters[2] >= print_control['c']:

                        reset_counter(counters, 2);
                        text_file_handles[3].truncate(0); text_file_handles[3].seek(0)
                        text_file_handles[3].write("#Time		#Step_1			#Step_2			#Step_3			#Step_4			#Step_5			#Step_6			#Step_7			#Step_interpolation	#Step_move_mesh		#Step_remeshing\n")
                        text_file_handles[3].write(f"{t:0,.10G}		{s1:0,.10G}		{s2:0,.10G}		{s3:0,.10G}		{s4:0,.10G}		{s5:0,.10G}		{s6:0,.10G}		{s7:0,.10G}		{si:0,.10G}		{sm:0,.10G}		{sr:0,.10G}\n\n")
                        if t < 0.98*T: text_file_handles[3].write("\n")

                    s_dt += timer_dt.stop()

                    # ----------------------------------

                    # Check for killvanDANA file
                    if os.path.isfile(path.join(result_folder.folder, "NSwithDLM")) == True:
                        print(RED % "--- killing NSwithDLM solver --- t = {}".format(t), "\n", flush = True)
                        break

        # ================================

        except Exception as e: print(BLUE % 'error message : ', flush = True); traceback.print_exc(file=sys.stdout) #; print(e, flush = True)

        finally:

            if t >= T:
                print(BLUE % '\nNS with DLM solver - COMPLETED : t = {}'.format(t), "\n", flush = True)
                complete = io.TextIOWrapper(open(result_folder.folder + "complete", "wb", 0), write_through=True)
                complete.seek(0); complete.write("{}, T = {}".format("COMPLETED", T))
                complete.close()
            
            """
            memory('Final memory use')
            print(RED % 'Total memory usage of solver = {} MB (RSS)'.format(str(memory.memory - initial_memory_use)), "\n", flush = True)
            """

            wall_time = timer_total.stop()

            text_file_handles[3].write("{} {} {}".format("\n\n", "DOFs -->", json.dumps(DOFS)))
            text_file_handles[3].write("{} {} {} {}".format("\n\n", "Total simulation wall time : ", wall_time, " sec"))
            text_file_handles[3].write("\n")

            print(RED % "Total simulation wall time : {} sec".format(wall_time), "\n", flush = True)

            for x in text_file_handles:
                x.close()
            for y,z in hdf5_file_handles.items():
                z.close(); del z

            if problem_physics['solve_FSI'] == True:
                read_time_series(Mpi.mpi_comm, curr_dir)

            if restart == True:
                extract_hdf5_data_for_xdmf_visualization(Mpi.mpi_comm, curr_dir, result_folder.bool_stream, problem_physics, fem_degree)

        # ================================


    def update_solid(self, mesh, t, dt):
        us_ = self.variables['us_']   
        
        # Sostituzione di Expression con l'estrazione geometrica esplicita in Firedrake
        X = SpatialCoordinate(mesh)
        displ_x = (self.amplitude * 0.5 * (1 - math.cos(0.4 * PI * t)))
        displ_y = 0.0    

    def assemble_tentative_velocity(self, u_, p_, Lm_f, dt):
        d = self.matrix
        u_ab = self.u_ab

        A1 = Matrix(self.A1)

        for ui in range(self.u_components):
            u_ab[ui].assign(0.0)
            u_ab[ui].vector().axpy(1.5, u_[1][ui].vector())
            u_ab[ui].vector().axpy(-0.5, u_[2][ui].vector())

        d['Cij'] = assemble(self.Cij)
        A1.axpy(-0.5, d['Cij'])

        X1 = Matrix(A1)
        X1.axpy(-2.0, d['Kij'])
        b1 = [None] * self.u_components
        for ui in range(self.u_components):
            b1[ui] = Fluid_problem.optimized_rhs(self, ui, X1, u_[1], p_[1])	
        
        A1.axpy(1.0, d['Cij'])

        for ui in range(self.u_components):
            b1[ui].axpy(1.0, d['Yij'][ui] * Lm_f.sub(ui).vector())

        return A1, b1
    
    def solve_tentative_velocity(self, A, x, b, bcs):
        for ui in range(self.u_components):
            solve(A, x[ui], b[ui], bcs=bcs[ui], solver_parameters=self.u_solver_params)

    def assemble_velocity_correction_DLM(self, u_, Lm_, Lm_old_, dt):
        bDLM = [None] * self.u_components
        for ui in range(self.u_components):
            bDLM[ui] = self.matrix['Mij'] * u_[ui].vector()
            bDLM[ui].axpy(float(dt), assemble(Lm_[ui] * self.v * self.dx - Lm_old_[ui] * self.v * self.dx))
        return bDLM

    def solve_velocity_correction(self, x, b, bcs):
        A = self.A3
        for ui in range(self.u_components):
            solve(A, x[ui], b[ui], bcs=bcs[ui], solver_parameters=self.u_c_solver_params)

    def assemble_lagrange_multiplier(self, Lm_, us_, uf_, dt):
        e = self.e; Lm = self.Lm; dx = self.dx
        A = assemble(dot(Lm, e) * dx)
        b = assemble((1 / dt) * dot(us_ - uf_, e) * dx + dot(Lm_[1], e) * dx)
        return A, b

    def solve_lagrange_multiplier(self, A, x, b):
        # Parametri equivalenti a 'bicgstab' + 'sor' in PETSc
        solve(A, x, b, solver_parameters={'ksp_type': 'bicgstab', 'pc_type': 'sor'})

    def calc_vorticity_streamfunction(self, u, bcs):
        p = self.p; q = self.q; dx = self.dx
        vort = self.variables['vort']
        psi = self.variables['psi']

        if self.bool_stream:
            a = p * q * dx
            L = (u[0].dx(1) - u[1].dx(0)) * q * dx
            solve(a == L, vort, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

            a = inner(grad(p), grad(q)) * dx
            L = vort * q * dx
            solve(a == L, psi, bcs=bcs, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

        return vort, psi
        

if __name__ == '__main__':

    # parsing arguments from command line
    parser = argparse.ArgumentParser(description = 'to append arguments from terminal')

    parser.add_argument('-velocity_degree', type=int, metavar='', required=False, default=fem_degree["velocity_degree"])
    parser.add_argument('-displacement_degree', type=int, metavar='', required=False, default=fem_degree["displacement_degree"])

    # arguments are stored in "args"
    args = parser.parse_args()

    # --------------------------------

    NS_DLM_Solver(args)

