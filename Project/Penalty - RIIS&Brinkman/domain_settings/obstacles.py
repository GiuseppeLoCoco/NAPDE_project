from __future__ import print_function
from firedrake import *

import numpy as np
from math import pi as PI

# Define circular obstacle, its signed distance function, and feature functions for Brinkman (in the circle) and RIIS (on the circumference)
class circleObstacle:

  # Create obstacle
  def __init__(self, x, y, r, riis_epsilon=0.05):
    # initial position
    self.x = x
    self.y = y
    self.r = r

    # RIIS half-thickness
    self.eps = riis_epsilon

    # displacement
    self.amplitude = 12*self.r
    self.displ_x = f'({self.amplitude}*0.5*(1 - cos(0.2*{PI}*t)))'
    self.displ_y = '0'
    self.us_x = f'({self.amplitude}*0.1*{PI}*sin(0.2*{PI}*t))'
    self.us_y = '0'

  # Signed distance function
  # def dist(self, x):
  #   return np.sqrt((x[0]-(self.x+self.displ_x(self.t)))**2 + (x[1]-(self.y+self.displ_y(self.t)))**2) - self.r
  def distStr(self):
    return f'sqrt( (x[0]-({self.x}+'+self.displ_x+f'))*(x[0]-({self.x}+'+self.displ_x+f')) + (x[1]-({self.y}+'+self.displ_y+f'))*(x[1]-({self.y}+'+self.displ_y+f')) ) - {self.r}'
  def distExpr(self, t=0):
    return Expression(self.distStr(),degree=2, t=t)
  def distFun(self, t=0):
    return interpolate(self.distExpr(t),FunctionSpace(mesh,'P',2))

  # Brinkman characteristic function
  def chi(self, t=0):
    return Expression('('+self.distStr()+') < 0', degree=2, t=t)

  # RIIS delta function
  def delta(self, t=0):
    return Expression(f'(1+cos({PI}*('+self.distStr()+f')/{self.eps}))/(2*{self.eps}) * (abs('+self.distStr()+f') < {self.eps})', degree=2, t=t)


# Define segment obstacle (codimension=1), its absolute distance function, and feature functions for RIIS
class lineObstacle:

  # Create obstacle
  def __init__(self, xA, yA, xB, yB, riis_epsilon):
    # initial position
    self.A_init = [xA, yA]
    self.B_init = [xB, yB]
    self.length = np.linalg.norm(np.array(self.A_init)-np.array(self.B_init))

    # RIIS half-thickness
    self.eps = riis_epsilon

    # displacement
    self.amplitude = 3*self.length
    self.displ_x = f'({self.amplitude}*0.5*(1-cos(2*{PI}*t)))'
    self.displ_y = '0'
    self.A = [f'({self.A_init[0]}+{self.displ_x})', f'({self.A_init[1]}+{self.displ_y})']
    self.B = [f'({self.B_init[0]}+{self.displ_x})', f'({self.B_init[1]}+{self.displ_y})']
    self.us_x = f'({self.amplitude}*{PI}*sin(2*{PI}*t))'
    self.us_y = '0'

  # Positive distance function
  def distStr(self):
    # First, we identify the projection of a generic point 'x' onto the line of the segment [A, B] -> 'abscissa' is its parametric coordinate, with abscissa=0,1 for A,B, respectively.
    abscissa = f'( (x[0]-{self.A[0]})*({self.B[0]}-{self.A[0]}) + (x[1]-{self.A[1]})*({self.B[1]}-{self.A[1]}) ) / {self.length**2}'
    # Then, we put thresholds on the abscissa.
    abscissa_thresh = f'( ({abscissa}) < 0 ? 0 : ( ({abscissa}) > 1 ? 1 : ({abscissa}) ) )'
    # Finally, the closest point to 'x' on the segment is A+abscissa_thresh*(B-A).   (see note above on the displacement)
    point_x = f'( {self.A[0]} + {abscissa_thresh}*(({self.B[0]}-{self.A[0]})) )'
    point_y = f'( {self.A[1]} + {abscissa_thresh}*(({self.B[1]}-{self.A[1]})) )'
    return f'sqrt( (x[0]-{point_x})*(x[0]-{point_x}) + (x[1]-{point_y})*(x[1]-{point_y}) )'
  def distExpr(self, t=0):
    return Expression(self.distStr(),degree=2, t=t)
  def distFun(self, t=0):
    return interpolate(self.distExpr(t),FunctionSpace(mesh,'P',2))

  # RIIS delta function
  def delta(self, t=0):
    return Expression(f'(1+cos({PI}*('+self.distStr()+f')/{self.eps}))/(2*{self.eps}) * (abs('+self.distStr()+f') < {self.eps})', degree=2, t=t)


# Segment rotating counterclockwise around its endpoint A.
class rotatingLineObstacle(lineObstacle):
  # Create obstacle
  def __init__(self, xA, yA, xB, yB, riis_epsilon):
    super().__init__(xA, yA, xB, yB, riis_epsilon)
    self.theta_max = PI/4.0
    theta = f'( 0-0.5*(1-cos(2*{PI}*t))*{self.theta_max} )'
    dottheta = f'( -{PI}*sin(2*{PI}*t)*{self.theta_max} )'
    self.A = self.A_init
    self.B = [ f'( {self.A[0]} + ({self.B_init[0]}-{self.A[0]})*cos({theta}) - ({self.B_init[1]}-{self.A[1]})*sin({theta}) )',
              f'( {self.A[1]} + ({self.B_init[0]}-{self.A[0]})*sin({theta}) + ({self.B_init[1]}-{self.A[1]})*cos({theta}) )' ]
    self.us_x = f'( {dottheta} * ( -({self.B_init[0]}-{self.A[0]})*sin({theta}) - ({self.B_init[1]}-{self.A[1]})*cos({theta}) ) )'
    self.us_y = f'( {dottheta} * ( ({self.B_init[0]}-{self.A[0]})*cos({theta}) - ({self.B_init[1]}-{self.A[1]})*sin({theta}) ) )'
