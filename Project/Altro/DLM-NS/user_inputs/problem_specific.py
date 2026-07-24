from firedrake import *
import sys, os

Lsc = characteristic_scales['Lsc']
Vsc = characteristic_scales['Vsc']
Tsc = Lsc / Vsc

perf = 0
blood_perfusion = False

if blood_perfusion:
    perf = 0.85 / 60 						
    perf *= (0.14 * Tsc)					

# In Firedrake creiamo una costante mutabile a runtime per gestire il tempo variabile
t_param = Constant(0.0)

# Profili definiti tramite combinazioni UFL native basate su SpatialCoordinate della mesh di riferimento
# NOTA: Quando applichi queste funzioni, assicurati di estrarre X = SpatialCoordinate(mesh)
def get_parabolic_profile(mesh):
    X = SpatialCoordinate(mesh)
    return 6.0 * X[1] * (4.1 - X[1]) / (4.1 * 4.1)

def get_inflow_profile(mesh, t_var):
    X = SpatialCoordinate(mesh)
    # Profilo dipendente dal tempo passato tramite Constant
    return 6.0 * X[1] * (4.1 - X[1]) / (4.1 * 4.1) * sin(2.0 * PI * t_var / 10.0)

def get_alternating_inflow_profile(mesh):
    X = SpatialCoordinate(mesh)
    return 0.3 * 4.0 * X[1] * (0.41 - X[1]) / (0.41 * 0.41)

# Per mantenere compatibilità con la chiamata originale a runtime nel loop temporale:
class TimeVaryingProfiles:
    def __init__(self, mesh):
        self.t = Constant(0.0)
        self.mesh = mesh
        
    def update_time(self, new_t):
        self.t.assign(new_t)

# Mock segnaposto per mantenere la consistenza sintattica se invocati altrove
parabolic_profile = 0.0
inflow_profile = 0.0

def time_varying_bc(tt):
    # Gestito direttamente aggiornando il valore dei parametri `Constant(t)` inseriti nelle forme variazionali
    pass

# Al posto di UserExpression, creiamo un'assegnazione spaziale condizionale basata sulla mappa dei subdomains
def get_shear_modulus_field(V_space, subdomains, Mat_0, Mat_1):
    # subdomains deve essere una Function discretizzata sullo stesso mesh (es: DG0)
    # Restituisce una funzione discontinua proiettata
    return project(conditional(eq(subdomains, 2), Mat_0, Mat_1), V_space)