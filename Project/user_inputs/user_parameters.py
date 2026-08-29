# ---- Geometry and mesh parameters ----

# Domain dimensions (channel)
Lx = 4
Ly = 1

# Line Obstacle parameters
xA = 0.5
xB = 2.0
yA = 0.5 * Ly
yB = 0.5 * Ly
line_thickness = 0.02

# Discretization parameter: number of elements
n = 150
n_conforming = 150

# Cylinder and Square Obstacle parameters
x_obs = 0.5
y_obs = 0.5 * Ly
r_obs = 0.1
side_length = 0.2

# Default flow parameters
Re = 40.0
R = 10000.0

# ---- Finite Element Spaces ----

fem_degree = dict(
    velocity_degree=2,
    pressure_degree=1,
    displacement_degree=1,
    lagrange_degree=1
)