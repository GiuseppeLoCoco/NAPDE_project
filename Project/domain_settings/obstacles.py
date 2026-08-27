from ast import Constant

from firedrake import *
import numpy as np
from math import pi as PI
import math # For Python math functions
# ==========================
# Define circular obstacle
# ==========================
class circleObstacle:
  def __init__(self, x, y, r, riis_epsilon=0.05):
    self.x_obs = x
    self.y_obs = y
    self.r = r
    self.eps = riis_epsilon
    self.amplitude = 12 * self.r

  def displ_x(self, t):
    from firedrake import cos, pi
    return self.amplitude * 0.5 * (1.0 - cos(0.2 * pi * t))

  def displ_y(self, t):
    return Constant(0.0)

  def us_x(self, t):
    from firedrake import sin, pi
    return self.amplitude * 0.1 * pi * sin(0.2 * pi * t)

  def us_y(self, t):
    return Constant(0.0)

  def distExpr(self, mesh, t):
    X = SpatialCoordinate(mesh)
    return sqrt((X[0] - (self.x_obs + self.displ_x(t)))**2 + (X[1] - (self.y_obs + self.displ_y(t)))**2) - self.r

  def chi(self, mesh, t):
    dist = self.distExpr(mesh, t)
    return conditional(dist < 0, 1.0, 0.0)

  def delta(self, mesh, t):
    from firedrake import pi, cos
    dist = self.distExpr(mesh, t)
    abs_dist = conditional(dist < 0.0, -dist, dist)
    return conditional(abs_dist < self.eps, (1.0 + cos(pi * dist / self.eps)) / (2.0 * self.eps), 0.0)

  def displacement(self, X0, t):
    return as_vector([self.displ_x(t), self.displ_y(t)])

  def velocity(self, X0, t):
    return as_vector([self.us_x(t), self.us_y(t)])

  def us_field(self, mesh, t):
    return as_vector([self.us_x(t), self.us_y(t)])

  def get_characteristic_length(self):
        return 2.0 * self.r


# ==========================
# Define segment obstacle
# ==========================
class lineObstacle:
  def __init__(self, xA, yA, xB, yB, riis_epsilon=0.05, thickness=0.01):
    self.A_init = [xA, yA] # Initial coordinates of point A
    self.B_init = [xB, yB]
    self.length = np.linalg.norm(np.array(self.A_init)-np.array(self.B_init))
    self.eps = riis_epsilon
    self.thickness = thickness
    self.amplitude = 3 * self.length

  def displ_x(self, t):
    from firedrake import cos, pi
    return self.amplitude * 0.5 * (1.0 - cos(2.0 * pi * t))
  
  def displ_y(self, t):
    return Constant(0.0)
    
  def A(self, t):
    return [self.A_init[0] + self.displ_x(t), self.A_init[1] + self.displ_y(t)]

  def B(self, t):
    return [self.B_init[0] + self.displ_x(t), self.B_init[1] + self.displ_y(t)]

  def us_x(self, t):
    from firedrake import sin, pi
    return self.amplitude * pi * sin(2.0 * pi * t)

  def us_y(self, t):
    return Constant(0.0)

  def distExpr(self, mesh, t):
    X = SpatialCoordinate(mesh)
    A_t = self.A(t)
    B_t = self.B(t)
    abscissa = ((X[0] - A_t[0]) * (B_t[0] - A_t[0]) + (X[1] - A_t[1]) * (B_t[1] - A_t[1])) / (self.length**2)
    abscissa_thresh = conditional(abscissa < 0, 0.0, conditional(abscissa > 1, 1.0, abscissa))
    point_x = A_t[0] + abscissa_thresh * (B_t[0] - A_t[0])
    point_y = A_t[1] + abscissa_thresh * (B_t[1] - A_t[1])
    return sqrt((X[0] - point_x)**2 + (X[1] - point_y)**2)

  def chi(self, mesh, t):
    dist = self.distExpr(mesh, t)
    return conditional(dist < 0, 1.0, 0.0)

  def delta(self, mesh, t):
    from firedrake import pi, cos
    dist = self.distExpr(mesh, t)
    abs_dist = conditional(dist < 0.0, -dist, dist)
    return conditional(abs_dist < self.eps, (1.0 + cos(pi * dist / self.eps)) / (2.0 * self.eps), 0.0)

  def displacement(self, X0, t):
    return as_vector([self.displ_x(t), self.displ_y(t)])

  def velocity(self, X0, t):
    return as_vector([self.us_x(t), self.us_y(t)])

  def us_field(self, mesh, t):
    return as_vector([self.us_x(t), self.us_y(t)])

  def get_characteristic_length(self):
        return self.length

  def get_gmsh_rectangle_params(self, t_val):
    """
    Returns parameters for Gmsh to create an unrotated thin rectangle
    and its rotation center and angle. For a non-rotating line, the angle is always 0.
    (xmin, ymin, dx, dy, rotation_center_x, rotation_center_y, angle)
    """
    x_center = (self.A_init[0] + self.B_init[0]) / 2.0
    y_center = (self.A_init[1] + self.B_init[1]) / 2.0
    
    xmin = x_center - self.length / 2.0
    ymin = y_center - self.thickness / 2.0
    dx = self.length
    dy = self.thickness
    
    # No rotation for the simple line obstacle
    return xmin, ymin, dx, dy, x_center, y_center, 0.0

# ------- Segment rotating counterclockwise --------
class rotatingLineObstacle(lineObstacle):
  def __init__(self, xA, yA, xB, yB, riis_epsilon=0.05, thickness=0.02):
    super().__init__(xA, yA, xB, yB, riis_epsilon, thickness)
    self.theta_max = PI / 4.0

  def theta(self, t):
    from firedrake import cos, pi
    return -0.5 * (1.0 - cos(2.0 * pi * t)) * self.theta_max
  
  def dottheta(self, t):
    from firedrake import sin, pi
    return -pi * sin(2.0 * pi * t) * self.theta_max

  def B(self, t):
    from firedrake import sin, cos
    th = self.theta(t)
    dx = self.B_init[0] - self.A_init[0]
    dy = self.B_init[1] - self.A_init[1]
    return [self.A_init[0] + dx * cos(th) - dy * sin(th),
            self.A_init[1] + dx * sin(th) + dy * cos(th)]

  def displacement(self, X0, t):
    from firedrake import sin, cos, as_vector
    th = self.theta(t)
    xA, yA = self.A_init[0], self.A_init[1]
    dX0 = X0[0] - xA
    dY0 = X0[1] - yA
    dx = dX0 * (cos(th) - 1.0) - dY0 * sin(th)
    dy = dX0 * sin(th) + dY0 * (cos(th) - 1.0)
    return as_vector([dx, dy])

  def velocity(self, X0, t):
    from firedrake import sin, cos, as_vector
    th = self.theta(t)
    dt = self.dottheta(t)
    xA, yA = self.A_init[0], self.A_init[1]
    dX0 = X0[0] - xA
    dY0 = X0[1] - yA
    us_x = dt * (-dX0 * sin(th) - dY0 * cos(th))
    us_y = dt * ( dX0 * cos(th) - dY0 * sin(th))
    return as_vector([us_x, us_y])

  def us_field(self, mesh, t):
    from firedrake import SpatialCoordinate, as_vector
    X = SpatialCoordinate(mesh)
    dt = self.dottheta(t)
    xA, yA = self.A_init[0], self.A_init[1]
    return as_vector([-dt * (X[1] - yA), dt * (X[0] - xA)])
            
  def us_x(self, t):
    from firedrake import sin, cos
    th = self.theta(t)
    dt = self.dottheta(t)
    dx = self.B_init[0] - self.A_init[0]
    dy = self.B_init[1] - self.A_init[1]
    return dt * (-dx * sin(th) - dy * cos(th))

  def us_y(self, t):
    from firedrake import sin, cos
    th = self.theta(t)
    dt = self.dottheta(t)
    dx = self.B_init[0] - self.A_init[0]
    dy = self.B_init[1] - self.A_init[1]
    return dt * (dx * cos(th) - dy * sin(th))

  def get_gmsh_rectangle_params(self, t_val):
    th = float(self.theta(t_val))
    x_center = (self.A_init[0] + self.B_init[0]) / 2.0
    y_center = (self.A_init[1] + self.B_init[1]) / 2.0
    
    xmin = x_center - self.length / 2.0
    ymin = y_center - self.thickness / 2.0
    dx = self.length
    dy = self.thickness
    
    return xmin, ymin, dx, dy, self.A_init[0], self.A_init[1], th




# ==========================
# Define square obstacle
# ==========================
class squareObstacle:
  def __init__(self, x, y, side_length=1.0, riis_epsilon=0.05):
      self.x_obs = x
      self.y_obs = y
      self.side_length = side_length
      self.half_side = side_length / 2.0
      self.eps = riis_epsilon

  def displ_x(self, t):
      return Constant(0.0)

  def displ_y(self, t):
      return Constant(0.0)

  def us_x(self, t):
      return Constant(0.0)

  def us_y(self, t):
      return Constant(0.0)

  def distExpr(self, mesh, t):

      X = SpatialCoordinate(mesh)
        
      # Center Position at time t
      xc = self.x_obs + self.displ_x(t)
      yc = self.y_obs + self.displ_y(t)

      dx_abs = conditional(X[0] - xc < 0, xc - X[0], X[0] - xc) - self.half_side
      dy_abs = conditional(X[1] - yc < 0, yc - X[1], X[1] - yc) - self.half_side

      d_out_x = conditional(dx_abs > 0.0, dx_abs, 0.0)
      d_out_y = conditional(dy_abs > 0.0, dy_abs, 0.0)
      dist_ext = sqrt(d_out_x**2 + d_out_y**2)

      max_d = conditional(dx_abs > dy_abs, dx_abs, dy_abs)
      dist_int = conditional(max_d < 0.0, max_d, 0.0)

      return dist_ext + dist_int

      """
      X = SpatialCoordinate(mesh)
      # Distanze dai bordi lungo x e y
      dx = conditional(X[0] - self.x_obs < 0, self.x_obs - X[0], X[0] - self.x_obs) - self.half_side
      dy = conditional(X[1] - self.y_obs < 0, self.y_obs - X[1], X[1] - self.y_obs) - self.half_side
      
      # Max(dx, dy) per la distanza (SDF) dal quadrato
      return conditional(dx > dy, dx, dy)
      """

  def chi(self, mesh, t):
      dist = self.distExpr(mesh, t)
      return conditional(dist < 0, 1.0, 0.0)

  def delta(self, mesh, t):
      from firedrake import pi, cos
      dist = self.distExpr(mesh, t)
      abs_dist = conditional(dist < 0.0, -dist, dist)
      return conditional(abs_dist < self.eps, (1.0 + cos(pi * dist / self.eps)) / (2.0 * self.eps), 0.0)

  def displacement(self, X0, t):
      return Constant((0.0, 0.0))

  def velocity(self, X0, t):
      return Constant((0.0, 0.0))

  def us_field(self, mesh, t):
      return Constant((0.0, 0.0))

  def get_characteristic_length(self):
        return self.side_length


# ==========================
# Define buffer layer obstacle
# ==========================
class BufferObstacle:
    """Obstacle representing the upstream buffer layer region (x < 0)."""
    def __init__(self, L_buf: float = 1.0, riis_epsilon: float = 0.05):
        self.L_buf = L_buf
        self.eps = riis_epsilon

    def chi(self, mesh, t=None):
        X = SpatialCoordinate(mesh)
        return conditional(lt(X[0], 0.0), 1.0, 0.0)

    def distExpr(self, mesh, t=None):
        X = SpatialCoordinate(mesh)
        return X[0]

    def delta(self, mesh, t=None):
        from firedrake import pi, cos
        dist = self.distExpr(mesh, t)
        abs_dist = conditional(dist < 0.0, -dist, dist)
        eps = getattr(self, 'eps', 0.05)
        return conditional(abs_dist < eps, (1.0 + cos(pi * dist / eps)) / (2.0 * eps), 0.0)

    def displacement(self, X0, t):
        return Constant((0.0, 0.0))

    def velocity(self, X0, t):
        return Constant((0.0, 0.0))

    def us_field(self, mesh, t):
        return Constant((0.0, 0.0))

    def us_x(self, t=None):
        return Constant(0.0)

    def us_y(self, t=None):
        return Constant(0.0)

    def get_characteristic_length(self):
        return 1.0