import sys
import os
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_settings')))

from firedrake import *
import argparse
from math import cos, pi as PI

from user_inputs import *
import user_inputs.user_parameters as user_parameters
from domain_settings import create_bcs_penalty, time_varying_bc, create_fluid_mesh
from obstacles import circleObstacle, squareObstacle, lineObstacle, rotatingLineObstacle
from post_processing import save_VTK, save_checkpoint, plot_results, create_output_folders
from Solvers.Stokes_solver import solve_stokes_initial

class Brinkman_solver:

    def __init__(self, moving=False, type_obstacle="square", n=None, R=None, Re=None, structured=False):

        self.moving = moving
        self.mean = True
        self.type_obstacle = type_obstacle
        self.n = n if n is not None else user_parameters.n
        self.R = R if R is not None else getattr(user_parameters, 'R', 10000.0)
        self.Re = Re if Re is not None else getattr(user_parameters, 'Re', 40.0)
        self.structured = structured
        self.symmetric = abs(y_obs - 0.5 * Ly) < 1e-6

    def Brinkman_solve(self, args=None, mesh=None, obstacle=None, f_custom=None, u_exact=None, p_exact=None, g_custom=None, u_init=None, dt=None, t_final=None):

        # start total timer
        t_start = time()

        # ==================================
        # CREATE MESH & OBSTACLE
        # ==================================
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            if self.type_obstacle not in ["line", "rotating", "rotating_line"]:
                if self.type_obstacle == "cylinder":
                    print("\nObstacle: Cylinder")
                    self.obstacle = circleObstacle(x_obs, y_obs, r_obs)
                elif self.type_obstacle == "square":
                    print("\nObstacle: Square")
                    self.obstacle = squareObstacle(x_obs, y_obs, side_length)

                if y_obs == Ly/2:
                    print("\nSymmetric configuration: cylinder centered in the channel")
                    self.symmetric = True
                else:
                    print("\nAsymmetric configuration: cylinder moved higher in the channel")
                    self.symmetric = False
            else:
                self.symmetric = False
                if self.type_obstacle == "line":
                    print("\nObstacle: Line")
                    self.obstacle = lineObstacle(xA, yA, xB, yB, thickness=line_thickness)
                else:
                    print("\nObstacle: Rotating Line")
                    self.obstacle = rotatingLineObstacle(xA, yA, xB, yB, thickness=line_thickness)

        if mesh is None:
            if self.structured:
                mesh = RectangleMesh(self.n, int(self.n * Ly / Lx), Lx, Ly)
            else:
                mesh = create_fluid_mesh(Lx, Ly, self.n, structured=False).mesh

        # ==================================
        # DATA AND SOLVER
        # ==================================

        tol = 1e-10

        T_end = float(t_final) if t_final is not None else 20.0
        dt = float(dt) if dt is not None else 0.5
        num_steps = max(1, int(round(T_end / dt)))
        
        # Reynolds number
        Re = self.Re

        # Density   
        rho = 1.0  

        # Characteristic velocity
        u_char = 1              # mean velocity

        # Charateristic length
        L_char = self.obstacle.get_characteristic_length()

        # Dynamic viscosity
        mu = rho * L_char * u_char / Re

        print(f"\nCharacteristic length L_char = {L_char}")
        print(f"\nReynolds number Re = {Re} computed with u_characteristic = {u_char}\n")

        f = f_custom(mesh) if callable(f_custom) else (f_custom if f_custom is not None else Constant((0, 0)))
        u_ex_val = u_exact(mesh) if callable(u_exact) else u_exact
        p_ex_val = p_exact(mesh) if callable(p_exact) else p_exact
        g_ex_val = g_custom(mesh) if callable(g_custom) else g_custom
        t = Constant(0.0)

        R = self.R


        # Define function spaces
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q

        # Boundary tag mapping (both RectangleMesh and Gmsh):
        # 1 = Left (Inflow), 2 = Right (Outflow), 3 = Bottom, 4 = Top
        dirichlet_tags = (1, 3, 4)
        outflow_tag = 2

        # Define boundary conditions
        if u_ex_val is not None:
            if g_ex_val is None:
                bcs = [DirichletBC(W.sub(0), u_ex_val, "on_boundary")]
                if p_ex_val is not None:
                    bcs.append(DirichletBC(W.sub(1), p_ex_val, "on_boundary"))
            else:
                bcs = [DirichletBC(W.sub(0), u_ex_val, tag) for tag in dirichlet_tags]
        else:
            bcs = create_bcs_penalty(W, mesh, type_obstacle=self.type_obstacle)

        # Define trial and test functions
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        # Define functions for solutions at previous and current time steps
        uh_n = Function(V)
        sol = Function(W)
        uh, ph = sol.subfunctions

        # Define expressions for Brinkman
        phi_expr = self.obstacle.distExpr(mesh, t)
        chi_expr = self.obstacle.chi(mesh, t)
        if u_ex_val is not None:
            us_expr = u_ex_val
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_field'):
            us_expr = self.obstacle.us_field(mesh, t)
        elif self.obstacle is not None and hasattr(self.obstacle, 'us_x'):
            us_expr = as_vector((self.obstacle.us_x(t), self.obstacle.us_y(t)))
        else:
            us_expr = Constant((0.0, 0.0))


        w = Constant((0.0, 0.0))

        # ==================================
        # DEFINE VARIATIONAL PROBLEM
        # ==================================

        a = Constant(rho)/Constant(dt)*inner(u, v)*dx \
              + Constant(rho)*inner(dot(uh_n - w, nabla_grad(u)), v)*dx \
              + 2.0 * Constant(mu)*inner(sym(grad(u)), sym(grad(v)))*dx \
              - div(v)*p*dx \
              + div(u)*q*dx \
              + Constant(R) * inner(u, v) * chi_expr * dx
        
        L = Constant(rho)/Constant(dt)*inner(uh_n, v)*dx \
              + inner(f, v)*dx \
              + Constant(R) * inner(us_expr, v)* chi_expr * dx

        if g_ex_val is not None:
            ds_b = Measure("ds", domain=mesh)
            L += inner(g_ex_val, v)*ds_b(outflow_tag)


        # =========================================
        # Create the folder
        # =========================================
        params = {
            'moving': self.moving,
            'obstacle': self.type_obstacle,
            'symmetric': self.symmetric,
            'n': self.n,
            'R': R,
            'Re': Re,
            'is_mms': (u_exact is not None),
            'structured': getattr(self, 'structured', True)
        }
        basedir, file_dict = create_output_folders('Brinkman', params, extra_fields=['phi', 'chi'])

        # Time-stepping
        t_val = 0.0
        time_varying_bc(0.0)

        # Initial condition initialization for velocity at t = 0
        if u_init is not None:
            if callable(u_init):
                uh_n.interpolate(u_init(mesh))
            else:
                uh_n.assign(u_init)
        else:
            print("Initializing velocity with stationary Stokes solver (t=0)...")
            uh_stokes, _ = solve_stokes_initial(
                mesh=mesh, bcs=bcs, mu=mu, f_custom=f_custom, g_custom=g_custom, W=W
            )
            uh_n.assign(uh_stokes)

        uh.assign(uh_n)

        DG1 = FunctionSpace(mesh, 'DG', 1)
        phiFun = Function(DG1)
        chiFun = Function(DG1)
        phiFun.interpolate(phi_expr)
        chiFun.interpolate(chi_expr)

        save_VTK(file_dict, t_val, uh, ph, phi=phiFun, chi=chiFun)
        
        save_checkpoint(basedir, t_val, mesh, self.moving, velocity=uh, pressure=ph, phi=phiFun, chi=chiFun)

        for step in range(num_steps):
            # Update current time
            t_val += dt
            print('t =', t_val)
            t.assign(t_val)
            time_varying_bc(t_val)
    
            solve(a == L, sol, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})

            phiFun.interpolate(phi_expr)
            chiFun.interpolate(chi_expr)
            
            save_VTK(file_dict, t_val, uh, ph, phi=phiFun, chi=chiFun)

            # Update previous solution
            uh_n.assign(uh)

            # Print max velocity
            print('\tu_max:', uh.dat.data.max())
            
            save_checkpoint(basedir, t_val, mesh=None, moving=self.moving, velocity=uh, pressure=ph, phi=phiFun, chi=chiFun)
            plot_results(mesh, uh, ph, t_val=t_val, basedir=basedir)

        wall_time = time() - t_start

        print('Total wall time = {} seconds'.format(wall_time), "\n", flush = True)
        return mesh, uh, ph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stokes Brinkman solver script')
    parser.add_argument('--moving', action='store_true', default=True, help='Use moving obstacle')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'square', 'line', 'rotating', 'rotating_line'],
                        help='Type of obstacle to use in the simulation.')
    parser.add_argument('--structured', action='store_true', default=False,
                        help='Use structured Cartesian fluid mesh')
    args = parser.parse_args()

    # Istanziamo la classe e chiamiamo il solver
    solver = Brinkman_solver(moving=args.moving, type_obstacle=args.obstacle, structured=args.structured)
    solver.Brinkman_solve()


