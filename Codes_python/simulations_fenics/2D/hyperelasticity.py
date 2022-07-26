import dolfin as df
import matplotlib.pyplot as plt
import multiphenics as mph
import time
import mshr
import vedo
import vedo.dolfin as vdf
import numpy as np

degV = 2
degPhi = degV + 1

# Function used to write in the outputs files
def output_latex(file, A, B):
    for i in range(len(A)):
        file.write("(")
        file.write(str(A[i]))
        file.write(",")
        file.write(str(B[i]))
        file.write(")\n")
    file.write("\n")


test_case = "circle"

if test_case == "circle":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 / 8.0 + (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.5

    def neumann(point):
        return point.x() < 0.5

    def interface(point):
        return df.near(point.x(), 0.5)

elif test_case == "ellipse":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.0

    def neumann(point):
        return point.x() < 0.0

    def interface(point):
        return df.near(point.x(), 0.0)


material = "NeoHookean"  # SaintVenant NeoHookean Gent

if material == "SaintVenant":

    # Define constants
    young, nu = 5000.0, 0.4
    mu, lmbda = (
        df.Constant(young / (2 * (1 + nu))),
        df.Constant(young * nu / ((1 + nu) * (1 - 2 * nu))),
    )

    def tensors(u):
        # Define strain measures
        I = df.Identity(len(u))  # the identity matrix
        F = df.variable(I + df.grad(u))  # the deformation gradient
        C = F.T * F  # the right Cauchy-Green deformation tensor
        E = 0.5 * (C - I)  # the Green-Lagrange strain tensor

        # Define strain energy density
        E = df.variable(E)
        W = (
            lmbda / 2 * (df.tr(E) ** 2)
            + mu * df.tr(E * E)
            + 10.0 * (df.det(F) - 1.0) ** 2
        )  # Saint-Venant Kirchhoff

        # Define Piola-Kirchoff stress tensors
        S = df.diff(W, E)  # the second Piola-Kirchoff stress tensor
        P = df.diff(W, F)  # the first Piola-Kirchoff stress tensor

        return P

elif material == "NeoHookean":
    # Define constants
    young, nu = df.Constant(5000.0), df.Constant(0.4)
    mu, lmbda = (
        df.Constant(young / (2.0 * (1.0 + nu))),
        df.Constant(young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))),
    )

    def tensors(u):
        # Define strain measures
        I = df.Identity(len(u))  # the identity matrix
        F = df.variable(I + df.grad(u))  # the deformation gradient
        C = F.T * F  # the right Cauchy-Green deformation tensor
        # E = 0.5*(C - I)      # the Green-Lagrange strain tensor

        # Define strain energy density
        # E = df.variable(E)
        I_C = df.tr(C)
        J = df.det(F)
        W = (
            (mu / 2.0) * (I_C - 3.0)
            - mu * df.ln(J)
            + (lmbda / 2.0) * (df.ln(J)) ** 2
            + 10.0 * (J - 1.0) ** 2
        )  # NeoHookean
        # Define Piola-Kirchoff stress tensors
        # S = df.diff(W, E) # the second Piola-Kirchoff stress tensor
        P = df.diff(W, F)  # the first Piola-Kirchoff stress tensor
        return P

elif material == "Gent":
    young, nu = df.Constant(5000.0), df.Constant(0.4)
    mu, lmbda = (
        df.Constant(young / (2.0 * (1.0 + nu))),
        df.Constant(young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))),
    )
    E = mu
    Jm = 13.0

    def tensors(u):
        d = u.geometric_dimension()
        I = df.Identity(d)  # Identity tensor
        F = df.variable(I + df.grad(u))  # Deformation gradient
        C = df.variable(F.T * F)  # Right Cauchy-Green tensor
        J = df.variable(df.det(F))
        I1 = df.variable(df.tr(C))
        J1 = df.variable(I1 * J ** (-2 / 3.0))
        W = (-E / 6.0) * Jm * df.ln(1.0 - (J1 - 3.0) / Jm) + 10.0 * (J - 1.0) ** 2
        P = df.diff(W, F)
        return P


hh_phi, error_l2_phi, error_h1_phi = [], [], []
error_l2_standard, hh_standard, error_h1_standard = [], [], []

start, end, step = 1, 4, 1
if test_case == "ellipse":
    start, end = start + 1, end + 1
for i in range(start, end, step):
    H = 8 * 2**i
    if test_case == "circle":
        background_mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(1.0, 1.0), H, H)
    elif test_case == "ellipse":
        background_mesh = df.RectangleMesh(
            df.Point(-1.0, -1.0), df.Point(1.0, 1.0), H, H
        )

    print("\nPhi FEM iteration: ", i)
    print("Nb of vertices", background_mesh.num_vertices())

    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )

    print("start cell selection")
    ls_time = time.time()
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        for v in df.vertices(cell):
            if phi(v.point()) <= 0.0 or df.near(phi(v.point()), 0.0):
                Cell_omega[cell] = 1
                break

    ls_time = time.time() - ls_time
    print("end cell election: ", ls_time)

    ls_time = time.time()
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    ls_time = time.time() - ls_time
    print("end submesh creation: ", ls_time)

    print("beginning of selection of the cells")
    # Selection of cells and facets on the boundary
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    cell_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
    facet_dirichlet_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 1, False
    )
    vertices_dirichlet_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 2, False
    )

    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1, False)
    vertices_neumann_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 2, False
    )

    Neumann, Dirichlet, Interface = 1, 2, 3
    for cell in df.cells(mesh):  # all triangles
        for facet in df.facets(cell):  # all faces
            v1, v2 = df.vertices(facet)  # all vertices of a face
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):

                # Cells for dirichlet condition
                if dirichlet(v1.point()) and dirichlet(v2.point()):
                    Cell[cell] = Dirichlet
                    cell_dirichlet_sub[cell] = 1
                    for facett in df.facets(cell):
                        Facet[facett] = Dirichlet
                        facet_dirichlet_sub[facett] = 1
                        for vert in df.vertices(facett):
                            vertices_dirichlet_sub[vert] = 1
                # Cells for Neumann condition
                else:
                    Cell[cell] = Neumann
                    cell_neumann_sub[cell] = 1
                    for facett in df.facets(cell):
                        Facet[facett] = Neumann
                        facet_neumann_sub[facett] = 1
                        for vert in df.vertices(facett):
                            vertices_neumann_sub[vert] = 1
    for cell in df.cells(mesh):
        if Cell[cell] == 0:
            for facet in df.facets(cell):
                if Facet[facet] == Neumann:
                    Facet[facet] = 4

    File2 = df.File("sub_dirichlet.rtc.xml/mesh_function_2.xml")
    File2 << cell_dirichlet_sub

    File1 = df.File("sub_dirichlet.rtc.xml/mesh_function_1.xml")
    File1 << facet_dirichlet_sub

    File0 = df.File("sub_dirichlet.rtc.xml/mesh_function_0.xml")
    File0 << vertices_dirichlet_sub

    yp_dirichlet_res = mph.MeshRestriction(mesh, "sub_dirichlet.rtc.xml")

    File2 = df.File("sub_neumann.rtc.xml/mesh_function_2.xml")
    File2 << cell_neumann_sub

    File1 = df.File("sub_neumann.rtc.xml/mesh_function_1.xml")
    File1 << facet_neumann_sub

    File0 = df.File("sub_neumann.rtc.xml/mesh_function_0.xml")
    File0 << vertices_neumann_sub
    yp_neumann_res = mph.MeshRestriction(mesh, "sub_neumann.rtc.xml")

    # Spaces and expressions of f, u_ex and boundary conditions

    V = df.VectorFunctionSpace(mesh, "CG", degV, dim=2)
    u_ex = df.Expression(
        ("3.*cos(x[0]/90.0)*sin(x[1]/90.0)", "3.*sin(x[0]/90.0)*cos(x[1]/90.0)"),
        degree=3,
        domain=mesh,
    )

    Z_N = df.TensorFunctionSpace(mesh, "CG", degV, shape=(2, 2))
    Q_N = df.VectorFunctionSpace(mesh, "DG", degV - 1, 2)
    Q_D = df.VectorFunctionSpace(mesh, "DG", degV, 2)
    W = mph.BlockFunctionSpace(
        [V, Z_N, Q_N, Q_D],
        restrict=[None, yp_neumann_res, yp_neumann_res, yp_dirichlet_res],
    )

    uyp = mph.BlockFunction(W)
    (u, y, p_N, p_D) = mph.block_split(uyp)

    duyp = mph.BlockTrialFunction(W)
    (du, dy, dp_N, dp_D) = mph.block_split(duyp)

    vzq = mph.BlockTestFunction(W)
    (v, z, q_N, q_D) = mph.block_split(vzq)

    P_ex = tensors(u_ex)
    P_u = tensors(u)

    P_v = df.derivative(P_u, u, v)

    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)

    # Modification of the measures to consider cells and facets on Omega_h^Gamma for the additional terms
    dx = df.Measure("dx", mesh, subdomain_data=Cell)
    ds = df.Measure("ds", mesh, subdomain_data=Facet)
    dS = df.Measure("dS", mesh, subdomain_data=Facet)

    gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p = (
        10.0,
        1.0,
        0.01,
        0.01,
        200000.0,
        0.20,
        0.001,
    )

    # 10.0, 1.0, 0.01, 0.01, 200000.0, 0.20, 0.001

    # Construction of the bilinear and linear forms
    boundary_penalty = (
        sigma_N * df.avg(h) * df.inner(df.jump(P_u, n), df.jump(P_v, n)) * dS(Neumann)
        + sigma_D
        * df.avg(h)
        * df.inner(df.jump(P_u, n), df.jump(P_v, n))
        * dS(Dirichlet)
        + sigma_D * h**2 * (df.inner(df.div(P_u), df.div(P_v))) * dx(Dirichlet)
    )

    phi_abs = df.inner(df.grad(phi), df.grad(phi)) ** 0.5

    print("h=", mesh.hmax())
    max_charg = 1
    for ii in range(max_charg):
        g = (
            df.dot(P_ex, df.grad(phi)) / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
            + u_ex * phi
        )

        u_D = u_ex * (1.0 + phi)

        f = -df.div(P_ex)

        auv = (
            df.inner(P_u, df.grad(v)) * dx
            + gamma_u * df.inner(P_u, P_v) * dx(Neumann)
            + boundary_penalty
            + gamma_D * h ** (-2) * df.inner(u, v) * dx(Dirichlet)
            - df.inner(df.dot(P_u, n), v) * ds(Dirichlet)
        )

        auz = gamma_u * df.inner(P_u, z) * dx(Neumann)
        auq_N = 0.0
        auq_D = -gamma_D * h ** (-2) * df.dot(u, q_D * phi) * dx(Dirichlet)

        ayv = df.inner(df.dot(y, n), v) * ds(Neumann) + gamma_u * df.inner(y, P_v) * dx(
            Neumann
        )
        ayz = (
            gamma_u * df.inner(y, z) * dx(Neumann)
            + gamma_div * df.inner(df.div(y), df.div(z)) * dx(Neumann)
            + gamma_p
            * h ** (-2)
            * df.inner(df.dot(y, df.grad(phi)), df.dot(z, df.grad(phi)))
            * dx(Neumann)
        )
        ayq_N = (
            gamma_p
            * h ** (-3)
            * df.inner(df.dot(y, df.grad(phi)), q_N * phi)
            * dx(Neumann)
        )
        ayq_D = 0.0

        ap_Nv = 0.0
        ap_Nz = (
            gamma_p
            * h ** (-3)
            * df.inner(p_N * phi, df.dot(z, df.grad(phi)))
            * dx(Neumann)
        )
        ap_Nq_N = gamma_p * h ** (-4) * df.inner(p_N * phi, q_N * phi) * dx(Neumann)
        ap_Nq_D = 0.0

        ap_Dv = -gamma_D * h ** (-2) * df.dot(v, p_D * phi) * dx(Dirichlet)
        ap_Dz = 0.0
        ap_Dq_N = 0.0
        ap_Dq_D = gamma_D * h ** (-2) * df.inner(
            p_D * phi, q_D * phi
        ) * dx + sigma_p * df.avg(h) ** (-1) * df.inner(
            df.jump(p_D * phi), df.jump(q_D * phi)
        ) * dS(
            Dirichlet
        )

        lv = (
            df.inner(f, v) * dx
            + sigma_D * h**2 * df.inner(f, -df.div(P_v)) * dx(Dirichlet)
            + gamma_D * h ** (-2) * df.dot(u_D, v) * dx(Dirichlet)
        )

        lz = gamma_div * df.inner(f, df.div(z)) * dx(Neumann) - gamma_p * h ** (
            -2
        ) * df.inner(g * phi_abs, df.dot(z, df.grad(phi))) * dx(Neumann)
        lq_N = -gamma_p * h ** (-3) * df.inner(g * phi_abs, q_N * phi) * dx(Neumann)
        lq_D = -gamma_D * h ** (-2) * df.inner(u_D, q_D * phi) * dx(Dirichlet)

        F = [
            auv + ayv + ap_Nv + ap_Dv - lv,
            auz + ayz + ap_Nz + ap_Dz - lz,
            auq_N + ayq_N + ap_Nq_N + ap_Dq_N - lq_N,
            auq_D + ayq_D + ap_Nq_D + ap_Dq_D - lq_D,
        ]

        J = mph.block_derivative(F, uyp, duyp)

        snes_solver_parameters = {
            "nonlinear_solver": "snes",
            "snes_solver": {
                "linear_solver": "mumps",
                "maximum_iterations": 40,
                "report": True,
                "error_on_nonconvergence": False,
            },
        }

        problem = mph.BlockNonlinearProblem(F, uyp, None, J)
        solver = mph.BlockPETScSNESSolver(problem)
        solver.parameters.update(snes_solver_parameters["snes_solver"])

        start_solve = time.time()
        solver.solve()
        end_solve = time.time()

    u_h = uyp[0]
    hh_phi.append(mesh.hmax())
    # Compute and store relative error for H1 and L2 norms
    relative_error_L2_phi_fem = df.sqrt(
        df.assemble((df.inner(u_ex - u_h, u_ex - u_h) * df.dx))
    ) / df.sqrt(df.assemble((df.inner(u_ex, u_ex)) * df.dx))
    print("Relative error L2 phi FEM : ", relative_error_L2_phi_fem)
    error_l2_phi.append(relative_error_L2_phi_fem)
    relative_error_H1_phi_fem = df.sqrt(
        df.assemble((df.inner(df.grad(u_ex - u_h), df.grad(u_ex - u_h)) * df.dx))
    ) / df.sqrt(df.assemble((df.inner(df.grad(u_ex), df.grad(u_ex))) * df.dx))
    error_h1_phi.append(relative_error_H1_phi_fem)
    print("Relative error H1 phi FEM : ", relative_error_H1_phi_fem)

# Computation of the standard FEM
if test_case == "circle":
    domain = mshr.Circle(
        df.Point(0.5, 0.5), df.sqrt(2.0) / 4.0
    )  # creation of the domain
elif test_case == "ellipse":
    domain = mshr.Ellipse(df.Point(0, 0), 1.0 / 2.0, 1.0 / 3.0)
for i in range(start, end, step):
    H = 8 * 2 ** (
        i - 1
    )  # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain, H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.VectorFunctionSpace(mesh, "CG", degV, dim=2)

    # selection facet
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Facet.set_all(0)
    Neumann, Dirichlet = 1, 2
    for facet in df.facets(mesh):
        v1, v2 = df.vertices(facet)
        # Cells for dirichlet condition
        if dirichlet(v1.point()) and dirichlet(v2.point()):
            Facet[facet] = Dirichlet
        # Cells for Neumann condition
        else:
            Facet[facet] = Neumann
    ds = df.Measure("ds", mesh, subdomain_data=Facet)

    if test_case == "circle":
        boundary = "on_boundary && x[0] >= 0.5"
    elif test_case == "ellipse":
        boundary = "on_boundary && x[0] >= 0.0"

    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    u_ex = df.Expression(
        ("3.*cos(x[0]/90.0)*sin(x[1]/90.0)", "3.*sin(x[0]/90.0)*cos(x[1]/90.0)"),
        degree=3,
        domain=mesh,
    )
    u_D = u_ex * (1.0 + phi)
    bc = df.DirichletBC(V, u_D, boundary)
    v = df.TestFunction(V)
    u = df.Function(V)

    P_ex = tensors(u_ex)
    f = -df.div(P_ex)
    g = (
        df.dot(P_ex, df.grad(phi)) / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
        + u_ex * phi
    )
    P = tensors(u)

    # Define nonlinear problem
    F = (
        df.inner(P, df.grad(v)) * df.dx
        - df.dot(f, v) * df.dx
        - df.dot(g, v) * ds(Neumann)
    )
    snes_solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "linear_solver": "mumps",
            "maximum_iterations": 40,
            "report": True,
            "error_on_nonconvergence": False,
        },
    }

    df.solve(F == 0, u, bcs=bc, solver_parameters=snes_solver_parameters)
    u_h = u
    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    error_l2_standard.append(
        (df.assemble((((u_ex - u_h)) ** 2) * df.dx) ** (0.5))
        / (df.assemble((((u_ex)) ** 2) * df.dx) ** (0.5))
    )
    error_h1_standard.append(
        (df.assemble(((df.grad(u_ex - u_h)) ** 2) * df.dx) ** (0.5))
        / (df.assemble(((df.grad(u_ex)) ** 2) * df.dx) ** (0.5))
    )


file_name = "hyperelasticity"
#  Write the output file for latex
file = open(f"outputs/{test_case}/{file_name}_{material}_P{degV}.txt", "w")
file.write(
    f"gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p = {str([gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p])} \n"
)
file.write("relative L2 norm phi fem: \n")
output_latex(file, hh_phi, error_l2_phi)
file.write("relative H1 norm phi fem : \n")
output_latex(file, hh_phi, error_h1_phi)
file.write("relative L2 norm classic fem: \n")
output_latex(file, hh_standard, error_l2_standard)
file.write("relative H1 normclassic fem : \n")
output_latex(file, hh_standard, error_h1_standard)
file.close()
