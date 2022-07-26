import dolfin as df
import matplotlib.pyplot as plt
import mshr
import time
import multiphenics as mph


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
    H = 4 * 2**i
    if test_case == "sphere":
        background_mesh = df.BoxMesh(
            df.Point(0.0, 0.0, 0.0), df.Point(1.0, 1.0, 1.0), H, H, H
        )
    elif test_case == "ellipsoid":
        background_mesh = df.BoxMesh(
            df.Point(-1.0, -1.0, -1.0), df.Point(1.0, 1.0, 1.0), H, H, H
        )

    # We now define Omega using phi
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
    hh_phi.append(mesh.hmax())  # store the size of each element for this iteration

    # Creation of the FunctionSpace for Phi on the new mesh
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    # Selection of cells and facets on the boundary
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

    # vedo.show(df.SubMesh(mesh, Cell, Dirichlet))
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

    V = df.FunctionSpace(mesh, "CG", degV)
    u_ex = df.Expression("sin(x[0])*exp(x[1])*cos(2*pi*x[2])", degree=6, domain=mesh)

    g = (
        df.dot(df.grad(u_ex), df.grad(phi))
        / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
        + u_ex * phi
    )
    Z_N = df.VectorFunctionSpace(mesh, "CG", degV, dim=3)
    Q_N = df.FunctionSpace(mesh, "DG", degV - 1)
    Q_D = df.FunctionSpace(mesh, "DG", degV)

    W = mph.BlockFunctionSpace(
        [V, Z_N, Q_N, Q_D],
        restrict=[None, yp_neumann_res, yp_neumann_res, yp_dirichlet_res],
    )

    uyp = mph.BlockTrialFunction(W)
    (u, y, p_N, p_D) = mph.block_split(uyp)

    vzq = mph.BlockTestFunction(W)
    (v, z, q_N, q_D) = mph.block_split(vzq)

    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)

    # Modification of the measures to consider cells and facets on Omega_h^Gamma for the additional terms
    dx = df.Measure("dx", mesh, subdomain_data=Cell)
    ds = df.Measure("ds", mesh, subdomain_data=Facet)
    dS = df.Measure("dS", mesh, subdomain_data=Facet)

    # gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p = 1.0, 0.01, 20.0, 0.01, 20000.0, 0.020, 1.0

    # Construction of the bilinear and linear forms
    boundary_penalty = (
        sigma_N
        * df.avg(h)
        * df.inner(df.jump(df.grad(u), n), df.jump(df.grad(v), n))
        * dS(Neumann)
        + sigma_D
        * df.avg(h)
        * df.inner(df.jump(df.grad(u), n), df.jump(df.grad(v), n))
        * dS(Dirichlet)
        + sigma_D
        * h**2
        * (df.inner(df.div(df.rgad(u)), df.div(df.grad(v))))
        * dx(Dirichlet)
    )

    phi_abs = df.inner(df.grad(phi), df.grad(phi)) ** 0.5

    gamma_div, gamma_u, gamma_p, sigma_N, gamma_D, sigma_D, sigma_p = (
        1.0,
        0.01,
        20.0,
        0.01,
        20000.0,
        0.020,
        1.0,
    )

    u_D = u_ex * (1.0 + phi)

    f = -df.div(df.grad(u_ex))

    auv = (
        df.inner(df.grad(u), df.grad(v)) * dx
        + gamma_u * df.inner(df.grad(u), df.grad(v)) * dx(Neumann)
        + boundary_penalty
        + gamma_D * h ** (-2) * df.inner(u, v) * dx(Dirichlet)
        - df.inner(df.dot(df.grad(u), n), v) * ds(Dirichlet)
    )

    auz = gamma_u * df.inner(df.grad(u), z) * dx(Neumann)
    auq_N = 0.0
    auq_D = -gamma_D * h ** (-2) * df.dot(u, q_D * phi) * dx(Dirichlet)

    ayv = df.inner(df.dot(y, n), v) * ds(Neumann) + gamma_u * df.inner(
        y, df.grad(v)
    ) * dx(Neumann)
    ayz = (
        gamma_u * df.inner(y, z) * dx(Neumann)
        + gamma_div * df.inner(df.div(y), df.div(z)) * dx(Neumann)
        + gamma_p
        * h ** (-2)
        * df.inner(df.dot(y, df.grad(phi)), df.dot(z, df.grad(phi)))
        * dx(Neumann)
    )
    ayq_N = (
        gamma_p * h ** (-3) * df.inner(df.dot(y, df.grad(phi)), q_N * phi) * dx(Neumann)
    )
    ayq_D = 0.0

    ap_Nv = 0.0
    ap_Nz = (
        gamma_p * h ** (-3) * df.inner(p_N * phi, df.dot(z, df.grad(phi))) * dx(Neumann)
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
        + sigma_D * h**2 * df.inner(f, -df.div(df.grad(v))) * dx(Dirichlet)
        + gamma_D * h ** (-2) * df.dot(u_D, v) * dx(Dirichlet)
    )

    lz = gamma_div * df.inner(f, df.div(z)) * dx(Neumann) - gamma_p * h ** (
        -2
    ) * df.inner(g * phi_abs, df.dot(z, df.grad(phi))) * dx(Neumann)
    lq_N = -gamma_p * h ** (-3) * df.inner(g * phi_abs, q_N * phi) * dx(Neumann)
    lq_D = -gamma_D * h ** (-2) * df.inner(u_D, q_D * phi) * dx(Dirichlet)

    a = [
        [auv, auz, auq_N, auq_D],
        [ayv, ayz, ayq_N, ayq_D],
        [ap_Nv, ap_Nz, ap_Nq_N, ap_Nq_D],
        [ap_Dv, ap_Dz, ap_Dq_N, ap_Dq_D],
    ]

    l = [lv, lz, lq_N, lq_D]
    start_assemble = time.time()
    A = mph.block_assemble(a)
    B = mph.block_assemble(l)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble - start_assemble)
    UU = mph.BlockFunction(W)
    start_solve = time.time()
    mph.block_solve(A, UU.block_vector(), B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve - start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h = UU[0]
    # Compute and store relative error for H1 and L2 norms
    error_l2_phi.append(
        (df.assemble((((u_ex - u_h)) ** 2) * dx) ** (0.5))
        / (df.assemble((((u_ex)) ** 2) * dx) ** (0.5))
    )
    error_h1_phi.append(
        (df.assemble(((df.grad(u_ex - u_h)) ** 2) * dx) ** (0.5))
        / (df.assemble(((df.grad(u_ex)) ** 2) * dx) ** (0.5))
    )

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
    H = 4 * 2 ** (
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
    u = df.TrialFunction(V)
    u_ex = df.Expression("sin(x[0])*exp(x[1])*cos(2*pi*x[2])", degree=6, domain=mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)

    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = -df.div(df.grad(u_ex))
    g = (
        df.inner(df.grad(u_ex), df.grad(phi))
        / df.inner(df.grad(phi), df.grad(phi)) ** (0.5)
        + u_ex * phi
    )
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u), df.grad(v)) * df.dx
    l = f * v * df.dx + g * v * ds(Neumann)
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(l)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble - start_assemble)
    start_standard = time.time()
    u = df.Function(V)
    bc.apply(A, B)
    df.solve(A, u.vector(), B)
    end_standard = time.time()
    u_h = u
    Time_solve_standard.append(end_standard - start_standard)
    Time_total_standard.append(Time_assemble_standard[-1] + Time_solve_standard[-1])
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

file_name = "poisson_neumann"
#  Write the output file for latex
file = open(f"outputs/{test_case}/{file_name}_P{degV}.txt", "w")
file.write("relative L2 norm phi fem: \n")
output_latex(file, hh_phi, error_l2_phi)
file.write("relative H1 norm phi fem : \n")
output_latex(file, hh_phi, error_h1_phi)
file.write("relative L2 norm and time phi fem : \n")
output_latex(file, error_l2_phi, Time_total_phi)
file.write("relative H1 norm and time phi fem : \n")
output_latex(file, error_h1_phi, Time_total_phi)
file.write("relative L2 norm classic fem: \n")
output_latex(file, hh_standard, error_l2_standard)
file.write("relative H1 normclassic fem : \n")
output_latex(file, hh_standard, error_h1_standard)
file.write("relative L2 norm and time classic fem : \n")
output_latex(file, error_l2_standard, Time_total_standard)
file.write("relative H1 norm and time classic fem : \n")
output_latex(file, error_h1_standard, Time_total_standard)
file.close()
