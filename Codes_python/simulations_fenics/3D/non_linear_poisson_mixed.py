import dolfin as df
import matplotlib.pyplot as plt
import multiphenics as mph
import mshr
import time

import ufl

# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"


# degree of interpolation for V and Vphi
degV = 2
degPhi = 1 + degV

# Function used to write in the outputs files
def output_latex(file, A, B):
    for i in range(len(A)):
        file.write("(")
        file.write(str(A[i]))
        file.write(",")
        file.write(str(B[i]))
        file.write(")\n")
    file.write("\n")


test_case = "ellipsoid"  # "sphere"

if test_case == "sphere":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = (
                -1.0 / 8.0 + (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2
            )

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.5

    def neumann(point):
        return point.x() < 0.5

    def interface(point):
        return df.near(point.x(), 0.5)

elif test_case == "ellipsoid":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2 + (x[2] * 3.0) ** 2

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.0

    def neumann(point):
        return point.x() < 0.0

    def interface(point):
        return df.near(point.x(), 0.0)


# We create the lists that we'll use to store errors and computation time for the phi-fem and standard fem
(
    Time_assemble_phi,
    Time_solve_phi,
    Time_total_phi,
    error_l2_phi,
    error_h1_phi,
    hh_phi,
) = ([], [], [], [], [], [])
(
    Time_assemble_standard,
    Time_solve_standard,
    Time_total_standard,
    error_h1_standard,
    error_l2_standard,
    hh_standard,
) = ([], [], [], [], [], [])


# we compute the phi-fem for different sizes of cells
start, end, step = 0, 5, 1
for i in range(start, end, step):
    print("Phi-fem iteration : ", i)

    # we define parameters and the "global" domain O
    H = 8 * 2**i
    if test_case == "sphere":
        background_mesh = df.BoxMesh(
            df.Point(0.0, 0.0, 0.0), df.Point(1.0, 1.0, 1.0), H, H, H
        )
    elif test_case == "ellipsoid":
        background_mesh = df.BoxMesh(
            df.Point(-1.0, -1.0, -1.0), df.Point(1.0, 1.0, 1.0), H, H, H
        )

    # Creation of Omega_h
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim(), 0
    )

    for cell in df.cells(background_mesh):
        for v in df.vertices(cell):
            if phi(v.point()) <= 0.0 or df.near(phi(v.point()), 0.0):
                Cell_omega[cell] = 1
                break

    mesh = df.SubMesh(background_mesh, Cell_omega, 1)
    hh_phi.append(mesh.hmax())

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    # Selection of cells and facets on the boundary for Omega_h^Gamma
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    cell_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
    facet_dirichlet_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 1, False
    )
    edges_dirichlet_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 2, False
    )
    vertices_dirichlet_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 3, False
    )

    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1, False)
    edges_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2, False)
    vertices_neumann_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 3, False
    )

    Neumann, Dirichlet, Interface = 1, 2, 3
    for cell in df.cells(mesh):  # all tetras
        for facet in df.facets(cell):  # all triangles
            v1, v2, v3 = df.vertices(facet)  # all vertices of a quad
            if (
                phi(v1.point()) * phi(v2.point()) <= 0.0
                or df.near(phi(v1.point()) * phi(v2.point()), 0.0)
                or phi(v1.point()) * phi(v3.point()) <= 0.0
                or df.near(phi(v1.point()) * phi(v3.point()), 0.0)
                or phi(v2.point()) * phi(v3.point()) <= 0.0
                or df.near(phi(v2.point()) * phi(v3.point()), 0.0)
            ):

                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                Cell[cell] = Interface
                for facett in df.facets(cell):
                    Facet[facett] = Interface

                # Cells for dirichlet condition
                if (
                    dirichlet(v1.point())
                    and dirichlet(v2.point())
                    and dirichlet(v3.point())
                ):
                    Cell[cell] = Dirichlet
                    cell_dirichlet_sub[cell] = 1
                    for facett in df.facets(cell):
                        Facet[facett] = Dirichlet
                        facet_dirichlet_sub[facett] = 1
                        for edge in df.edges(facett):
                            edges_dirichlet_sub[edge] = 1
                        for vert in df.vertices(facett):
                            vertices_dirichlet_sub[vert] = 1
                # Cells for Neumann condition
                elif (
                    neumann(v1.point()) and neumann(v2.point()) and neumann(v3.point())
                ):
                    Cell[cell] = Neumann
                    cell_neumann_sub[cell] = 1
                    for facett in df.facets(cell):
                        Facet[facett] = Neumann
                        facet_neumann_sub[facett] = 1
                        for edge in df.edges(facett):
                            edges_neumann_sub[edge] = 1
                        for vert in df.vertices(facett):
                            vertices_neumann_sub[vert] = 1
    for cell in df.cells(mesh):
        if Cell[cell] == 0:
            for facet in df.facets(cell):
                if Facet[facet] == Neumann:
                    Facet[facet] = 4

    File3 = df.File("sub_dirichlet.rtc.xml/mesh_function_3.xml")
    File3 << cell_dirichlet_sub

    File2 = df.File("sub_dirichlet.rtc.xml/mesh_function_2.xml")
    File2 << facet_dirichlet_sub

    File1 = df.File("sub_dirichlet.rtc.xml/mesh_function_1.xml")
    File1 << edges_dirichlet_sub

    File0 = df.File("sub_dirichlet.rtc.xml/mesh_function_0.xml")
    File0 << vertices_dirichlet_sub

    yp_dirichlet_res = mph.MeshRestriction(mesh, "sub_dirichlet.rtc.xml")

    File3 = df.File("sub_neumann.rtc.xml/mesh_function_3.xml")
    File3 << cell_neumann_sub

    File2 = df.File("sub_neumann.rtc.xml/mesh_function_2.xml")
    File2 << facet_neumann_sub

    File1 = df.File("sub_neumann.rtc.xml/mesh_function_1.xml")
    File1 << edges_neumann_sub

    File0 = df.File("sub_neumann.rtc.xml/mesh_function_0.xml")
    File0 << vertices_neumann_sub
    yp_neumann_res = mph.MeshRestriction(mesh, "sub_neumann.rtc.xml")

    # Spaces and expressions of f, u_ex and boundary conditions
    V = df.FunctionSpace(mesh, "CG", degV)
    u_ex = df.Expression("1 + x[0] + 2*x[1] + 4*x[2]", degree=6, domain=mesh)

    Z_N = df.VectorFunctionSpace(mesh, "CG", degV, dim=3)
    Q_N = df.FunctionSpace(mesh, "DG", degV - 1)
    Q_D = df.FunctionSpace(mesh, "DG", degV)
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

    gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p = (
        100.0,
        1.0,
        10.0,
        0.01,
        10000.0,
        1.0,
        0.01,
    )
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)

    # Modification of the measures to consider cells and facets on Omega_h^Gamma for the additional terms
    dx = df.Measure("dx", mesh, subdomain_data=Cell)
    ds = df.Measure("ds", mesh, subdomain_data=Facet)
    dS = df.Measure("dS", mesh, subdomain_data=Facet)

    # Construction of the bilinear and linear forms
    boundary_penalty = (
        sigma_N
        * df.avg(h)
        * df.inner(
            df.jump((1.0 + u * u) * df.grad(u), n),
            df.jump((1.0 + u * u) * df.grad(v), n),
        )
        * dS(Neumann)
        + sigma_D
        * df.avg(h)
        * df.inner(
            df.jump((1.0 + u * u) * df.grad(u), n),
            df.jump((1.0 + u * u) * df.grad(v), n),
        )
        * (dS(Dirichlet) + dS(Interface))
        + sigma_D
        * h**2
        * (df.div((1.0 + u * u) * df.grad(u)) * df.div((1.0 + u * u) * df.grad(v)))
        * dx(Dirichlet)
    )

    phi_abs = df.inner(df.grad(phi), df.grad(phi)) ** 0.5

    g = (
        df.inner((1.0 + u_ex * u_ex) * df.grad(u_ex), df.grad(phi))
        / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
        + u_ex * phi
    )
    u_D = u_ex * (1.0 + phi)
    f = -df.div((1.0 + u_ex * u_ex) * df.grad(u_ex))

    auv = (
        df.inner((1.0 + u * u) * df.grad(u), df.grad(v)) * dx
        + gamma_u
        * df.inner((1.0 + u * u) * df.grad(u), (1.0 + u * u) * df.grad(v))
        * dx(Neumann)
        + boundary_penalty
        + gamma_D * h ** (-2) * u * v * dx(Dirichlet)
        - df.inner((1.0 + u * u) * df.grad(u), n) * v * (ds(Dirichlet) + ds(Interface))
    )

    auz = gamma_u * df.inner((1.0 + u * u) * df.grad(u), z) * dx(Neumann)
    auq_N = 0.0
    auq_D = -gamma_D * h ** (-3) * u * q_D * phi * dx(Dirichlet)

    ayv = df.inner(y, n) * v * ds(Neumann) + gamma_u * df.inner(
        y, (1.0 + u * u) * df.grad(v)
    ) * dx(Neumann)
    ayz = (
        gamma_u * df.inner(y, z) * dx(Neumann)
        + gamma_div * df.div(y) * df.div(z) * dx(Neumann)
        + gamma_p
        * h ** (-2)
        * df.inner(y, df.grad(phi))
        * df.inner(z, df.grad(phi))
        * dx(Neumann)
    )
    ayq_N = gamma_p * h ** (-3) * df.inner(y, df.grad(phi)) * q_N * phi * dx(Neumann)
    ayq_D = 0.0

    ap_Nv = 0.0
    ap_Nz = gamma_p * h ** (-3) * p_N * phi * df.inner(z, df.grad(phi)) * dx(Neumann)
    ap_Nq_N = gamma_p * h ** (-4) * p_N * phi * q_N * phi * dx(Neumann)
    ap_Nq_D = 0.0

    ap_Dv = -gamma_D * h ** (-3) * v * p_D * phi * dx(Dirichlet)
    ap_Dz = 0.0
    ap_Dq_N = 0.0
    ap_Dq_D = gamma_D * h ** (-4) * p_D * phi * q_D * phi * dx + sigma_p * df.avg(
        h
    ) ** (-1) * df.inner(df.jump(p_D * phi), df.jump(q_D * phi)) * dS(Dirichlet)

    lv = (
        df.inner(f, v) * dx
        - sigma_D * h**2 * f * df.div((1.0 + u * u) * df.grad(v)) * dx(Dirichlet)
        + gamma_D * h ** (-2) * u_D * v * dx(Dirichlet)
    )

    lz = gamma_div * f * df.div(z) * dx(Neumann) - gamma_p * h ** (
        -2
    ) * g * phi_abs * df.inner(z, df.grad(phi)) * dx(Neumann)
    lq_N = -gamma_p * h ** (-3) * g * phi_abs * q_N * phi * dx(Neumann)
    lq_D = -gamma_D * h ** (-3) * u_D * q_D * phi * dx(Dirichlet)

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
            "maximum_iterations": 20,
            "report": True,
            "error_on_nonconvergence": False,
        },
    }

    start_solve = time.time()
    problem = mph.BlockNonlinearProblem(F, uyp, None, J)
    mat = problem.F()
    print(mat)

    solver = mph.BlockPETScSNESSolver(problem)
    solver.parameters.update(snes_solver_parameters["snes_solver"])
    end_solve = time.time()
    solver.solve()
    Time_solve_phi.append(end_solve - start_solve)
    Time_total_phi.append(Time_solve_phi[-1])
    u_h = uyp[0]

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
    print("h :", mesh.hmax())


# Computation of the standard FEM
if test_case == "ellipsoid":
    domain = mshr.Ellipsoid(
        df.Point(0, 0, 0), 1.0 / 2.0, 1.0 / 3.0, 1.0 / 3.0
    )  # creation of the domain
elif test_case == "sphere":
    domain = mshr.Sphere(
        df.Point(0.5, 0.5, 0.5), df.sqrt(2.0) / 4.0
    )  # creation of the domain

for i in range(start, end, step):
    H = 8 * 2 ** (
        i - 1
    )  # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain, H)
    print("Standard fem iteration : ", i)
    # FunctionSpace Pk
    V = df.FunctionSpace(mesh, "CG", degV)

    if test_case == "sphere":
        boundary = "on_boundary && x[0] >= 0.5"
    elif test_case == "ellipsoid":
        boundary = "on_boundary && x[0] >= 0.0"

    u_D = u_ex * (1 + phi)
    bc = df.DirichletBC(V, u_D, boundary)

    # selection facet
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Facet.set_all(0)
    Neumann, Dirichlet = 1, 2
    for facet in df.facets(mesh):
        v1, v2, v3 = df.vertices(facet)
        # Cells for dirichlet condition
        if dirichlet(v1.point()) and dirichlet(v2.point()) and dirichlet(v3.point()):
            Facet[facet] = Dirichlet
        # Cells for Neumann condition
        elif neumann(v1.point()) and neumann(v2.point()) and neumann(v3.point()):
            Facet[facet] = Neumann
    ds = df.Measure("ds", mesh, subdomain_data=Facet)

    v = df.TestFunction(V)
    u = df.Function(V)
    u_ex = df.Expression("1 + x[0] + 2*x[1] + 4*x[2]", degree=6, domain=mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)

    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = -df.div((1.0 + u_ex * u_ex) * df.grad(u_ex))
    g = (
        df.inner((1.0 + u_ex * u_ex) * df.grad(u_ex), df.grad(phi))
        / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
        + u_ex * phi
    )  # Resolution of the variationnal problem
    F = (
        df.inner((1 + u**2) * df.grad(u), df.grad(v)) * dx
        - f * v * dx
        - g * v * ds(Neumann)
    )

    snes_solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True,
            "error_on_nonconvergence": False,
        },
    }

    df.solve(F == 0, u, bc, solver_parameters=snes_solver_parameters)
    u_h = u

    # Compute and store h and L2 H1 errors
    hh_standard.append(mesh.hmax())
    error_l2_standard.append(
        (df.assemble((((u_ex - u)) ** 2) * df.dx) ** (0.5))
        / (df.assemble((((u_ex)) ** 2) * df.dx) ** (0.5))
    )
    error_h1_standard.append(
        (df.assemble(((df.grad(u_ex - u)) ** 2) * df.dx) ** (0.5))
        / (df.assemble(((df.grad(u_ex)) ** 2) * df.dx) ** (0.5))
    )

file_name = "poisson_non_linear"
#  Write the output file for latex
file = open(f"outputs/{test_case}/{file_name}_P{degV}.txt", "w")
file.write("relative L2 norm phi fem: \n")
output_latex(file, hh_phi, error_l2_phi)
file.write("relative H1 norm phi fem : \n")
output_latex(file, hh_phi, error_h1_phi)
file.write("relative L2 norm classic fem: \n")
output_latex(file, hh_standard, error_l2_standard)
file.write("relative H1 normclassic fem : \n")
output_latex(file, hh_standard, error_h1_standard)
file.close()
