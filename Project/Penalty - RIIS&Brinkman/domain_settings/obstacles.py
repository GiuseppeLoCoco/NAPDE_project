from firedrake import *
import numpy as np
from math import pi as PI

# Define circular obstacle
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
    from firedrake import pi, cos, abs
    dist = self.distExpr(mesh, t)
    return conditional(abs(dist) < self.eps, (1.0 + cos(pi * dist / self.eps)) / (2.0 * self.eps), 0.0)

# Define segment obstacle
class lineObstacle:
  def __init__(self, xA, yA, xB, yB, riis_epsilon=0.05):
    self.A_init = [xA, yA]
    self.B_init = [xB, yB]
    self.length = np.linalg.norm(np.array(self.A_init)-np.array(self.B_init))
    self.eps = riis_epsilon
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
    from firedrake import pi, cos, abs
    dist = self.distExpr(mesh, t)
    return conditional(abs(dist) < self.eps, (1.0 + cos(pi * dist / self.eps)) / (2.0 * self.eps), 0.0)

# Segment rotating counterclockwise
class rotatingLineObstacle(lineObstacle):
  def __init__(self, xA, yA, xB, yB, riis_epsilon=0.05):
    super().__init__(xA, yA, xB, yB, riis_epsilon)
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