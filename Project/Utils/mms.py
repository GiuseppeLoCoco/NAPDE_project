"""
Manufactured Solutions and Analytical Source Terms for Navier-Stokes MMS Tests.
Provides analytical velocity, pressure, momentum body forcing, and boundary traction.
"""

from firedrake import (
    SpatialCoordinate, as_vector, dot, grad, sym, div, nabla_grad,
    sin, cos, pi, Identity
)


class ManufacturedSolution:
    """Exact divergence-free analytical solution, body force, and traction for Navier-Stokes MMS."""
    def __init__(self, Lx: float = 4.0, Ly: float = 1.0, Re: float = 40.0, rho: float = 1.0, L_char: float = 1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.rho = rho
        self.u_char = 1.0
        self.L_char = L_char  # Characteristic scale matching channel/buffer (1.0)
        self.mu = self.rho * self.L_char * self.u_char / self.Re

    def u_exact(self, mesh):
        X = SpatialCoordinate(mesh)
        x, y = X[0], X[1]
        u_x = 1.0 + sin(pi * x / self.Lx) * sin(2.0 * pi * y / self.Ly)
        u_y = (self.Ly / (2.0 * self.Lx)) * cos(pi * x / self.Lx) * (cos(2.0 * pi * y / self.Ly) - 1.0)
        return as_vector([u_x, u_y])

    def p_exact(self, mesh):
        X = SpatialCoordinate(mesh)
        x, y = X[0], X[1]
        return sin(pi * x / self.Lx) * sin(pi * y / self.Ly)

    def f_forcing(self, mesh):
        """Analytical momentum source term: f = rho*(u.grad)u - div(2*mu*sym(grad(u))) + grad(p)."""
        X = SpatialCoordinate(mesh)
        u_ex = self.u_exact(mesh)
        p_ex = self.p_exact(mesh)
        adv = self.rho * dot(u_ex, nabla_grad(u_ex))
        diff = - div(2.0 * self.mu * sym(grad(u_ex)))
        press = grad(p_ex)
        return adv + diff + press

    def g_exact(self, mesh):
        """Analytical boundary traction on the right outlet (x = Lx): g = sigma(u, p) * n with n = (1, 0)."""
        u_ex = self.u_exact(mesh)
        p_ex = self.p_exact(mesh)
        n_out = as_vector([1.0, 0.0])
        sigma_ex = 2.0 * self.mu * sym(grad(u_ex)) - p_ex * Identity(2)
        return dot(sigma_ex, n_out)
