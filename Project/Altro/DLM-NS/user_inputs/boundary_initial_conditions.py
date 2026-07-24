from firedrake import *
from .user_parameters import problem_physics
from .problem_specific import *				
from math import pi as PI

# Condizioni al contorno per il fluido
def fluid_create_boundary_conditions(fluid_mesh, **V):
    # In Firedrake passiamo direttamente l'ID numerico del marker della superficie
    mesh = fluid_mesh.mesh
    X = SpatialCoordinate(mesh)
    
    # velocity (V['fluid'][0] è lo spazio vettoriale o componente per la velocità)
    bcu_left_x = DirichletBC(V['fluid'][0], parabolic_profile, 1)
    bcu_bottom_x = DirichletBC(V['fluid'][0], Constant(0.0), 2)
    bcu_top_x = DirichletBC(V['fluid'][0], Constant(0.0), 4)
    bcu_x = [bcu_left_x, bcu_bottom_x, bcu_top_x]

    bcu_left_y = DirichletBC(V['fluid'][0], Constant(0.0), 1)
    bcu_bottom_y = DirichletBC(V['fluid'][0], Constant(0.0), 2)
    bcu_top_y = DirichletBC(V['fluid'][0], Constant(0.0), 4)
    bcu_y = [bcu_left_y, bcu_bottom_y, bcu_top_y]

    bcu = [bcu_x, bcu_y]

    # pressure
    bcp_right = DirichletBC(V['fluid'][1], Constant(0.0), 3)
    bcp = [bcp_right]

    # Streamfunction (su tutto il boundary esterno)
    bcPSI = DirichletBC(V['fluid'][1], Constant(0.0), "on_boundary")

    bcs = dict(velocity=bcu, pressure=bcp, streamfunction=bcPSI)

    if problem_physics['solve_temperature']:
        bcT_left = DirichletBC(V['fluid_temp'][0], Constant(1.0), 1)
        bcT_top = DirichletBC(V['fluid_temp'][0], Constant(0.0), 4)
        bcT = [bcT_left, bcT_top]
        bcs.update(temperature=bcT)
			
    return bcs

def prescrKin_fluid_create_boundary_conditions(fluid_mesh, **V):
    bcu_inflow_x = DirichletBC(V['fluid'][0], inflow_profile, 1)
    bcu_walls_x = DirichletBC(V['fluid'][0], Constant(0.0), 3)
    bcu_inflow_y = DirichletBC(V['fluid'][0], Constant(0.0), 1)
    bcu_walls_y = DirichletBC(V['fluid'][0], Constant(0.0), 3)

    bcu = [[bcu_walls_x, bcu_inflow_x], [bcu_inflow_y, bcu_walls_y]]

    bcp_outflow = DirichletBC(V['fluid'][1], Constant(0.0), 2)
    bcp = [bcp_outflow]

    bcPSI = DirichletBC(V['fluid'][1], Constant(0.0), "on_boundary")

    return dict(velocity=bcu, pressure=bcp, streamfunction=bcPSI)

def solid_create_boundary_conditions(solid_mesh, boundaries, dt, **V):
    mesh = solid_mesh.mesh
    X = SpatialCoordinate(mesh)
    
    # Sostituzione di RegionOfInterest: identifichiamo la regione geometrica direttamente.
    # In Firedrake possiamo vincolare zone geometriche usando espressioni condizionali 
    # o integrando su marker specifici se la mesh ha sotto-domini fisici definiti.
    # Supponendo una sferoide/cilindro geometrico condizionale:
    cond_roi = sqrt((X[0] - 2.0)**2 + (X[1] - 2.0)**2) < 0.500001
    
    if not problem_physics['compressible_solid']:
        bcx_cylinder = DirichletBC(V['solid'][1].sub(0), Constant((0.0, 0.0)), "on_boundary") 
    else:
        bcx_cylinder = DirichletBC(V['solid'][0], Constant((0.0, 0.0)), "on_boundary")

    return [bcx_cylinder]  

def poisson_create_boundary_conditions(input_mesh, **V):
    bcu_all = DirichletBC(V['poisson'][0], Constant(2.0), 10)
    bcu_hom = [DirichletBC(V['poisson'][0], Constant(0.0), 10)]
    return dict(poisson=[bcu_all], poisson_hom=bcu_hom)

def heat_create_Tex(mesh, t_val=0.0):
    # Al posto di Expression, restituiamo espressioni simboliche UFL pronte per l'assegnazione
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    T_expr = cos(PI * t_val) * (1.0 + cos(2.0 * PI * (0.5 + r)))
    
    # Funzione caratteristica (1 se True, 0 se False)
    characteristic_expr = conditional((X[0]-0.5)**2 / 0.25**2 + (X[1]-0.5)**2 / 0.125**2 < 1.0, 1.0, 0.0)
    return T_expr, characteristic_expr
	
def heat_create_boundary_conditions(fluid_mesh, **V):
    return dict(temperature=[])

# Inizializzazioni dei vettori soluzione
def fluid_create_initial_conditions(u_, p_, T_):
    for i in range(3):
        u_[i][0].assign(0.0)
        u_[i][1].assign(0.0)
        p_[i].assign(0.0)

    for i in range(3):
        T_[i].assign(0.0)

def solid_create_initial_conditions(Dp_, mix, dt):
    # In Firedrake si usa .assign() per copiare funzioni o costanti negli spazi/sottospazi
    mix.sub(1).assign(0.0)
    Dp_[0].assign(0.0) 
    Dp_[1].assign(0.0) 
    Dp_[2].assign(0.0) 
    mix.sub(0).assign(0.0)

def poisson_create_initial_conditions(u_):
    u_.assign(0.0)

def heat_create_initial_conditions(T_, Lm_, Ts_):
    for i in range(2):
        T_[i].assign(0.0)
        Lm_[i].assign(0.0)
    Ts_.assign(0.0)