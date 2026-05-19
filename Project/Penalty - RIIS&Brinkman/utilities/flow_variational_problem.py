from firedrake import *
from ufl import tensors, nabla_div, nabla_grad
import numpy as np
import sys
import inspect

sys.path.insert(0,  '..')
from user_inputs import *
from utilities.read import *
from petsc4py import PETSc


PI = 3.14159265

class Fluid_problem:	
									
	def __init__(self, fluid_mesh, bool_stream):
		
		allowed_keys = inspect.signature(calc_non_dimensional_numbers).parameters.keys()                # Get the parameter names of the function
		filt_physical_parameters = {k: v for k, v in physical_parameters.items() if k in allowed_keys}       # Filter dicts to only include keys in the function's parameters
		Re, _, _, Fr = calc_non_dimensional_numbers(**filt_physical_parameters, **characteristic_scales)

		mesh = fluid_mesh.mesh
		dim = mesh.geometric_dimension()
		
		# Velocity components
		self.u_components = dim
		
		V  = FunctionSpace(mesh, 'CG', fem_degree['velocity_degree'])		  # Fluid velocity   
		Q  = FunctionSpace(mesh, 'CG', fem_degree['pressure_degree'])		  # Fluid pressure
		Z1 = VectorFunctionSpace(mesh, 'CG', fem_degree['lagrange_degree'])                                            # Lagrange multiplier

		# --------------------------------

		Vp = VectorFunctionSpace(mesh, 'CG', fem_degree['velocity_degree'])

		self.assigner_uv = None # We'll do simple interpolation or assignment

		# --------------------------------

		self.u1  = TrialFunction(V)
		self.v   = TestFunction(V)
		self.p   = TrialFunction(Q)
		self.q   = TestFunction(Q)
		self.Lm1 = [TrialFunction(Z1.sub(ui)) for ui in range(self.u_components)]

		variables = dict(); u_ = []; p_ = []
		for i in range(3):
			u_.insert(i, as_vector([Function(V) for ui in range(self.u_components)]))
			p_.insert(i, Function(Q))

		# PISO inner_loop variables
		self.u_inner = as_vector([Function(V) for ui in range(self.u_components)])
		self.p_inner = Function(Q)

		self.p_x = Function(Q)
		self.pvc_factor = 0.0
		if pressure_velocity_coupling == "IPCS":	self.pvc_factor = 1.0

		uv   = Function(Vp)		
		Lm_f = Function(Z1)
		Lm_f.assign(0.0)

		variables.update(u_=u_, uv=uv, p_=p_, Lm_f=Lm_f)	
			
		vort, psi = Function(Q), Function(Q)
		vort.assign(0.0); psi.assign(0.0)
		variables.update(vort=vort, psi=psi)

		# --------------------------------

		# Body force
		f = Constant((0,)*dim)
		
		# --------------------------------

		self.nx = as_vector([1.0, 0.0, 0.0][:dim])
		self.ny = as_vector([0.0, 1.0, 0.0][:dim])
		if dim == 3: self.nz = as_vector([0.0, 0.0, 1.0])

		self.rs = []
		self.residual = Function(V)
		
		self.f = f
		self.dim = dim
		self.variables = variables
		self.bool_stream = bool_stream
		self.h_f = CellDiameter(mesh)
		self.h_f_X = Function(FunctionSpace(mesh, 'CG', 1)).interpolate(self.h_f)
		vertex_values_h_f_X = self.h_f_X.dat.data
		vn = np.max(1.0/(vertex_values_h_f_X))
		self.VN_local = vn*vn
		
		self.F = [V, Q, Z1]
		self.dx = dx
		self.ds = ds
		self.n = FacetNormal(mesh)
		if dim == 2: self.tang = as_vector([self.n[1], -self.n[0]])

		self.Re = Constant(Re)
		self.Fr = Constant(Fr)

		# Convection matrix
		self.u_ab = as_vector([Function(V) for ui in range(self.u_components)])    

		# --------------------------------
		# Define solver parameters for firedrake
		self.tentative_velocity_solver_params = {
            'ksp_type': tentative_velocity_solver.get('solver_type', 'gmres'),
            'pc_type': tentative_velocity_solver.get('preconditioner_type', 'jacobi')
        }
		self.pressure_correction_solver_params = {
            'ksp_type': pressure_correction_solver.get('solver_type', 'cg'),
            'pc_type': pressure_correction_solver.get('preconditioner_type', 'hypre')
        }
		self.velocity_correction_solver_params = {
            'ksp_type': velocity_correction_solver.get('solver_type', 'cg'),
            'pc_type': velocity_correction_solver.get('preconditioner_type', 'jacobi')
        }

	def pre_assemble(self, bcs, dt):
		pass


	def residual_NS_equation(self, ui, u1, u2, u_ab, p_, Lm_f, f, dt):
			
		U = 0.5*(u1 + u2)	
		self.residual = (u1 - u2)/dt + dot(u_ab, nabla_grad(U)) + p_.dx(ui) - nabla_div((2/self.Re)*nabla_grad(U)) - f[ui] - Lm_f[ui]


	# Predict tentative velocity
	def assemble_tentative_velocity(self, u_, p_, Lm_f, dt):

		Re = self.Re; f = self.f
		u1 = self.u1; v = self.v; u_ab = self.u_ab
		dx = self.dx; ds = self.ds; h_f = self.h_f

		# Advecting velocity 
		for ui in range(self.u_components):
			u_ab[ui].assign(1.5 * u_[1][ui] - 0.5 * u_[2][ui])

		# Assembling linear systems on the fly in Firedrake
		a = dot(u1, v)*dx/Constant(dt) + dot((0.5/Re)*nabla_grad(u1), nabla_grad(v))*dx + 0.5*dot(dot(u_ab, nabla_grad(u1)), v)*dx
		
		A1 = a
		b1 = [None]*self.u_components
		for ui in range(self.u_components):
			L = dot(u_[1][ui], v)*dx/Constant(dt) - dot((0.5/Re)*nabla_grad(u_[1][ui]), nabla_grad(v))*dx - 0.5*dot(dot(u_ab, nabla_grad(u_[1][ui])), v)*dx
			L += dot(f[ui], v)*dx
			L -= self.pvc_factor * p_[1].dx(ui) * v * dx
			L += Lm_f.sub(ui) * v * dx
			b1[ui] = L
			
		# Stabilization terms	
		if stabilization_parameters.get('SUPG_NS', False):
			from .fem_stabilizations import tau, Pop
			tau_supg = tau(alpha, u_[1], h_f, Re, dt); operator_supg = Pop(u_ab, v)
			for ui in range(self.u_components):
				self.residual_NS_equation(ui, u1, u_[1][ui], u_ab, p_[1], Lm_f, f, dt)
				S1 = tau_supg*dot(operator_supg, self.residual)*dx
				A1 += lhs(S1)
				b1[ui] += rhs(S1)

		if stabilization_parameters.get('crosswind_NS', False):
			from .fem_stabilizations import tau_cw, Pop_CW
			self.rs.clear()
			for ui in range(self.u_components):
				self.residual_NS_equation(ui, u_[1][ui], u_[2][ui], u_ab, p_[1], Lm_f, f, dt)
				self.rs.append(self.residual)
			R = as_vector(self.rs)
			A1 += inner(tau_cw(C_cw, u_[1], h_f, Re, R)*Pop_CW(u_ab, u1), nabla_grad(v))*dx

		return A1, b1	        
	
	def change_initial_guess(self, u):
		for i in range(self.dim):
			u[i].assign(0.0)
		
	def solve_tentative_velocity(self, A, x, b, bcs):
		for ui in range(self.u_components):
			solve(A == b[ui], x[ui], bcs=bcs[ui], solver_parameters=self.tentative_velocity_solver_params)


	# Pressure correction
	def assemble_pressure_correction(self, u_, p_, Lm_f, dt):

		p = self.p; q = self.q; dx = self.dx; f = self.f
		h_f = self.h_f; u_ab = self.u_ab; Re = self.Re
	
		A = dot(nabla_grad(p), nabla_grad(q))*dx
		L2 = (-1/Constant(dt))*div(u_[0])*q*dx
		
		if stabilization_parameters.get('PSPG_NS', False):
			from .fem_stabilizations import tau
			tau_pspg = tau(alpha, u_[0], self.h_f, Re, dt); operator_pspg = nabla_grad(q); self.rs.clear()
			for ui in range(self.u_components):
				self.residual_NS_equation(ui, u_[0][ui], u_[1][ui], u_ab, p, Lm_f, f, dt)
				self.rs.append(self.residual)
			R = as_vector(self.rs)	
			S2 = tau_pspg*dot(operator_pspg, R)*dx
			A += lhs(S2)
			L2 += rhs(S2)
		
		L2 += self.pvc_factor * dot(nabla_grad(p_), nabla_grad(q))*dx

		return A, L2

	def solve_pressure_correction(self, A, x, b, bcs):
		nullspace = None
		if bcs == []:
			nullspace = VectorSpaceBasis(constant=True)
		solve(A == b, x, bcs=bcs, nullspace=nullspace, solver_parameters=self.pressure_correction_solver_params)

	# Velocity correction  -  PISO
	def assemble_velocity_correction(self, u_, p_0, p_1, dt):
		b3 = [None]*self.u_components
		self.p_x.assign(self.pvc_factor * p_1)
		for ui in range(self.u_components):
			b3[ui] = u_[ui] * self.v * dx - Constant(dt) * (p_0.dx(ui) - self.p_x.dx(ui)) * self.v * dx
		return b3

	def solve_velocity_correction(self, x, b, bcs):
		A = self.u1 * self.v * dx
		for ui in range(self.u_components):
			solve(A == b[ui], x[ui], bcs=bcs[ui], solver_parameters=self.velocity_correction_solver_params)


	# Velocity correction  -  DLM
	def assemble_velocity_correction_DLM(self, u_, Lm_, Lm_old_, dt):
		bDLM = [None]*self.u_components
		for ui in range(self.u_components):
			bDLM[ui] = u_[ui] * self.v * dx + Constant(dt) * (Lm_[ui] - Lm_old_[ui]) * self.v * dx
		return bDLM

	# Post-processing functions
	def post_process_data(self, Mpi, u, p, t, tsp, text_file_handles):
		Re = self.Re; n = self.n; ds = self.ds
		from .constitutive_eq import sigma

		# Compute drag, lift
		traction = -1*dot(sigma(Re, u, p), n)
		drag = assemble(dot(traction, self.nx)*ds(4))/(0.5*(PI)/4)
		lift = assemble(dot(traction, self.ny)*ds(4))/(0.5*(PI)/4)

		if Mpi.get_rank() == 0:
		    text_file_handles[0].write(f"{t:0,.10G}		{drag:0,.10G}		{lift:0,.10G}\n")    

	def calc_vorticity_streamfunction(self, u, bcs):
		p = self.p; q = self.q; dx = self.dx
		vort = self.variables['vort']; psi = self.variables['psi']

		if self.bool_stream == True:
			# Compute vorticity by L2 projection
			a = p*q*dx
			L = (u.sub(0).dx(1) - u.sub(1).dx(0))*q*dx
			solve(a == L, vort, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

			# Compute stream function : Laplacian(psi) = -vort
			a = inner(grad(p), grad(q))*dx
			L = vort*q*dx
			solve(a == L, psi, bcs=bcs, solver_parameters={'ksp_type': 'gmres', 'pc_type': 'hypre'})

		return vort, psi