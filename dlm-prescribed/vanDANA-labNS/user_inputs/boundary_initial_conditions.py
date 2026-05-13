from dolfin import DirichletBC, Constant, assign, Expression, interpolate, SubDomain, \
					MeshFunction, sqrt, DOLFIN_EPS, near
from .user_parameters import problem_physics
from .problem_specific import *				
from math import pi as PI

class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        return bool(x[2] < DOLFIN_EPS and x[2] > -DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - 1.0

constrained_domain = None

class RegionOfInterest(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-6
        return sqrt(((x[0] - 2.0)*(x[0] - 2.0)) + ((x[1] - 2.0)*(x[1] - 2.0))) < 0.5 + tol

class Point_pressure(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 4.2) and near(x[1], 5.) and near(x[2], 2.)

# Boundary conditions
def fluid_create_boundary_conditions(fluid_mesh, **V):

	boundaries = fluid_mesh.get_mesh_boundaries()

	# velocity
	bcu_left_x = DirichletBC(V['fluid'][0], parabolic_profile, boundaries, 1)
	bcu_bottom_x = DirichletBC(V['fluid'][0], Constant(0), boundaries, 2)
	bcu_top_x = DirichletBC(V['fluid'][0], Constant(0), boundaries, 4)
	bcu_x = [bcu_left_x, bcu_bottom_x, bcu_top_x]

	bcu_left_y = DirichletBC(V['fluid'][0], Constant(0), boundaries, 1)
	bcu_bottom_y = DirichletBC(V['fluid'][0], Constant(0), boundaries, 2)
	bcu_top_y = DirichletBC(V['fluid'][0], Constant(0), boundaries, 4)
	bcu_y = [bcu_left_y, bcu_bottom_y, bcu_top_y]

	bcu = [bcu_x, bcu_y]

	# pressure
	bcp_right = DirichletBC(V['fluid'][1], Constant(0), boundaries, 3)
	bcp = [bcp_right]

	# Streamfunction
	wall  = 'on_boundary'
	bcPSI = DirichletBC(V['fluid'][1], 0, wall)

	bcs = dict(velocity = bcu, pressure = bcp, streamfunction = bcPSI)

	if problem_physics['solve_temperature'] == True:
		# temperature
		bcT_left = DirichletBC(V['fluid_temp'][0], Constant(1), boundaries, 1)
		bcT_top = DirichletBC(V['fluid_temp'][0], Constant(0), boundaries, 4)
		bcT = [bcT_left, bcT_top]
		
		bcs.update(temperature = bcT)
			
	return bcs

def prescrKin_fluid_create_boundary_conditions(fluid_mesh, **V):

	boundaries = fluid_mesh.get_mesh_boundaries()

	
	# velocity
	bcu_inflow_x = DirichletBC(V['fluid'][0], inflow_profile, boundaries, 1)
	bcu_walls_x = DirichletBC(V['fluid'][0], Constant(0), boundaries, 3)
	bcu_inflow_y = DirichletBC(V['fluid'][0], Constant(0), boundaries, 1)
	bcu_walls_y = DirichletBC(V['fluid'][0], Constant(0), boundaries, 3)

	bcu = [[bcu_walls_x, bcu_inflow_x], [bcu_inflow_y, bcu_walls_y]]

	# pressure
	bcp_outflow = DirichletBC(V['fluid'][1], Constant(0), boundaries, 2)
	bcp = [bcp_outflow]

	# Streamfunction
	wall  = 'on_boundary'
	bcPSI = DirichletBC(V['fluid'][1], 0, wall)

	bcs = dict(velocity = bcu, pressure = bcp, streamfunction = bcPSI)

	return bcs

def solid_create_boundary_conditions(solid_mesh, boundaries, dt, **V):

	cylinder = 0; Complement_cylinder = 1         
	mesh_part = MeshFunction("size_t", solid_mesh.mesh, 0, Complement_cylinder)     
	RegionOfInterest().mark(mesh_part, cylinder)
	subdomainR = RegionOfInterest()

	# Note to self: Boundary conditions are for incremental displacement (delta D)

	# Solid
	if problem_physics['compressible_solid'] == False:
		bcx_cylinder = DirichletBC(V['solid'][1].sub(0), Constant((0, 0)), subdomainR) #, method="pointwise")
	elif problem_physics['compressible_solid'] == True:
	    bcx_cylinder = DirichletBC(V['solid'][0], Constant((0, 0)), subdomainR) #, method="pointwise")

	bcx = [bcx_cylinder]  
	return bcx    


def poisson_create_boundary_conditions(input_mesh, **V):

	boundaries = input_mesh.get_mesh_boundaries()

	# bcu_left = DirichletBC(V['poisson'][0], Constant(1.0), boundaries, 1)
	# bcu_bottom = DirichletBC(V['poisson'][0], Constant(0), boundaries, 2)
	# bcu_top = DirichletBC(V['poisson'][0], Constant(0), boundaries, 4)
	# bcu = [bcu_left, bcu_bottom, bcu_top]

	bcu_all = DirichletBC(V['poisson'][0], Constant(2.0), boundaries, 10)
	bcu = [bcu_all]

	# homogeneous BC, for incremental algorithm
	bcu_hom = [DirichletBC(V['poisson'][0], Constant(0.0), boundaries, 10)]

	bcs = dict(poisson = bcu, poisson_hom = bcu_hom)

	return bcs


class HeatPeriodicDomain(SubDomain):

    # Left and bottom boundaries are "target domain" G
    def inside(self, x, on_boundary):
        return on_boundary and (
            (near(x[0], 0) and not near(x[1], 1))
            or
            (near(x[1], 0) and not near(x[0], 1))
            )

    # Map right+top boundary (H) to left+bottom boundary (G)
    def map(self, x, y):
        if near(x[0], 1):
            y[0] = x[0] - 1
        else:
            y[0] = x[0]

        if near(x[1], 1):
            y[1] = x[1] - 1
        else:
            y[1] = x[1]

heat_constrained_domain = HeatPeriodicDomain()

def heat_create_Tex():
	# Returns exact T imposed on constraint and characteristic function of the constraint region
	return \
		Expression('cos(pi*t)* ( 1+cos(2*3.14*(0.5+sqrt(x[0]*x[0] + x[1]*x[1]))) )', degree=3, t=0, pi=PI),\
		Expression('(x[0]-0.5)*(x[0]-0.5)/(0.25*0.25) + (x[1]-0.5)*(x[1]-0.5)/(0.125*0.125) < 1', degree=3)
	# return \
	# 	Expression('x[0]*x[0]*x[0] - x[1]*x[1]*x[1]', degree=3),\
	# 	Expression('(x[0]-0.5)*(x[0]-0.5)/(0.25*0.25) + (x[1]-0.5)*(x[1]-0.5)/(0.125*0.125) < 1', degree=3)
	
def heat_create_boundary_conditions(fluid_mesh, **V):
	return dict(temperature = [])

	boundaries = fluid_mesh.get_mesh_boundaries()

	bcT1 = DirichletBC(V['temperature'][0], heat_create_Tex()[0], boundaries, 10)
	
	bcT = [bcT1]

	bcs = dict(temperature = bcT)
			
	return bcs


# Initial conditions
def fluid_create_initial_conditions(u_, p_, T_):

	# Velocity / pressure
	for i in range(3):
		u_[i][0].vector()[:] = 0.0
		u_[i][1].vector()[:] = 0.0
		# u_[i][2].vector()[:] = 0.0
		p_[i].vector()[:] = 0.0

	# Temperature
	for i in range(3):
		T_[i].vector()[:] = 0.0
	

def solid_create_initial_conditions(Dp_, mix, dt):
	
	# Solid pressure (only defined for incompressible solid)
	assign(mix.sub(1), interpolate(Constant(0), mix.sub(1).function_space().collapse()))

	# Cumulative displacement
	Dp_[0].vector()[:] = 0.0 

	# Incremental displacement (delta D)
	Dp_[1].vector()[:] = 0.0 # V_init*dt
	Dp_[2].vector()[:] = 0.0 # V_init*dt
	assign(mix.sub(0), interpolate(Expression(('0.0', '0.0'), degree = 2), mix.sub(0).function_space().collapse()))


def poisson_create_initial_conditions(u_):
	u_.vector()[:] = 0.0

def heat_create_initial_conditions(T_, Lm_, Ts_):
	for i in range(2):
		T_[i].vector()[:] = 0.0
		Lm_[i].vector()[:] = 0.0
	Ts_.vector()[:] = 0.0
