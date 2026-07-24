from firedrake import Constant, sqrt
from ufl import tensors

restart = False									

problem_physics = dict(
    solve_temperature=False,				
    solve_FSI=True,						
    compressible_solid=True,				
    solid_material='neoHookean',											
    viscous_dissipation=False,			
    body_force=False,														 
)
	
def f_dir(dim):									
    # Vettore unitario UFL
    n = -1 * tensors.unit_vector(1, dim) 
    return n

interpolation_fx = 'phi4'

stabilization_parameters = dict(	
    SUPG_NS=False,						
    PSPG_NS=False,								
    crosswind_NS=False,					
    SUPG_HT=False,						
    crosswind_HT=False					
)

alpha = Constant(0.85)                   	  	
C_cw = Constant(0.7)                       		

physical_parameters = dict(
    g=9.81,														  
    rho_f=1.0,								
    nu=0.001, 									
    Spht_f=1.0,         					
    K_f=1.0,								
    rho_s=10.0,								
    Sm=0.0,									
    Ld=0.0,									
    nw=0.4,								
    Spht_s=0.11,							
    K_s=1.2, 								
    heat_mu=0.1,
    heat_alpha=100.0
)

def calc_non_dimensional_solid_properties(g, rho_f, nu, Spht_f, K_f, rho_s, Sm, Ld, nw, Spht_s, K_s, Lsc, Vsc, T0, Tm, Tsc):
    rho = rho_s / rho_f						
    Spht = Spht_s / Spht_f					
    K = K_s / K_f								
    Ld = 2000.0 
    Nw = nw
    Sm = 500.0 
    return rho, Spht, K, Ld, Nw, Sm

characteristic_scales = dict(
    Lsc=1.0,			            		          
    Vsc=1.0,	         		    
    T0=-1.0 * 52.0,								
    Tm=37.0									
)

time_control = dict(
    C_no=1.0,								
    C_vi=10.0,								
    C_kn=10.0,								
    dt=0.1,  								
    tmax=5.0,								
    dt_min=0.0,						
    adjustable_timestep=False,			
    maxit=100, 								
    fd_tol=1e-8							
)
time_control['T'] = time_control['tmax']

fem_degree = dict(
    velocity_degree=2,
    pressure_degree=1,
    temperature_degree=1, 
    displacement_degree=1,
    lagrange_degree=1
)

def calc_non_dimensional_numbers(g, rho_f, nu, Spht_f, K_f, rho_s, Sm, Ld, nw, Spht_s, K_s, Lsc, Vsc, T0, Tm, Tsc):
    Re = 1.0 / nu 
    Pr = 1.0 
    Ec = 0.4 
    Fr = Vsc / sqrt(g * Lsc) 
    return Re, Pr, Ec, Fr

post_process = True

print_control = dict(
    a=1,   								
    b=20,  								
    c=20, 								
    d=5,   								
    e=2    								
)

calc_stream_function = True