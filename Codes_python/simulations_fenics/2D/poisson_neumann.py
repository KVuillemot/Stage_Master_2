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
degV = 1
degPhi = 2 + degV

# Function used to write in the outputs files
def output_latex(file, A, B):
    for i in range(len(A)):
        file.write("(")
        file.write(str(A[i]))
        file.write(",")
        file.write(str(B[i]))
        file.write(")\n")
    file.write("\n")


test_case = "ellipse"

if test_case == "circle":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 / 8.0 + (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2

        def value_shape(self):
            return (2,)

elif test_case == "ellipse":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2

        def value_shape(self):
            return (2,)


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
start, end, step = 1, 6, 1
if test_case == "ellipse":
    start, end = start + 1, end + 1
for i in range(start, end, step):
    print("Phi-fem iteration : ", i)
    # we define parameters and the "global" domain O
    H = 8 * 2**i
    if test_case == "circle":
        background_mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(1.0, 1.0), H, H)
    elif test_case == "ellipse":
        background_mesh = df.RectangleMesh(
            df.Point(-1.0, -1.0), df.Point(1.0, 1.0), H, H
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

    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1, False)
    vertices_neumann_sub = df.MeshFunction(
        "bool", mesh, mesh.topology().dim() - 2, False
    )

    Neumann = 1
    for cell in df.cells(mesh):  # all triangles
        for facet in df.facets(cell):  # all faces
            v1, v2 = df.vertices(facet)  # all vertices of a face
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):
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
                    Facet[facet] = 2

    File2 = df.File("sub_neumann.rtc.xml/mesh_function_2.xml")
    File2 << cell_neumann_sub

    File1 = df.File("sub_neumann.rtc.xml/mesh_function_1.xml")
    File1 << facet_neumann_sub

    File0 = df.File("sub_neumann.rtc.xml/mesh_function_0.xml")
    File0 << vertices_neumann_sub
    yp_neumann_res = mph.MeshRestriction(mesh, "sub_neumann.rtc.xml")

    # Variationnal problem resolution
    V = df.FunctionSpace(mesh, "CG", degV)
    Z = df.VectorFunctionSpace(mesh, "CG", degV, dim=mesh.topology().dim())
    Q = df.FunctionSpace(mesh, "DG", degV - 1)
    W = mph.BlockFunctionSpace(
        [V, Z, Q], restrict=[None, yp_neumann_res, yp_neumann_res]
    )
    uyp = mph.BlockTrialFunction(W)
    (u, y, p) = mph.block_split(uyp)
    vzq = mph.BlockTestFunction(W)
    (v, z, q) = mph.block_split(vzq)

    dx = df.Measure("dx", mesh, subdomain_data=Cell)
    ds = df.Measure("ds", mesh)
    dS = df.Measure("dS", mesh, subdomain_data=Facet)

    gamma_div, gamma_u, gamma_p, sigma = 1.0, 1.0, 1.0, 0.01
    h = df.CellDiameter(mesh)
    n = df.FacetNormal(mesh)
    u_ex = df.Expression("sin(x[0])*exp(x[1])", degree=6, domain=mesh)
    f = -df.div(df.grad(u_ex)) + u_ex
    g = (
        df.inner(df.grad(u_ex), df.grad(phi))
        / (df.inner(df.grad(phi), df.grad(phi)) ** 0.5)
        + u_ex * phi
    )

    # Construction of the bilinear and linear forms
    boundary_penalty = (
        sigma
        * df.avg(h)
        * df.inner(df.jump(df.grad(u), n), df.jump(df.grad(v), n))
        * dS(Neumann)
    )

    phi_abs = df.inner(df.grad(phi), df.grad(phi)) ** 0.5

    auv = (
        df.inner(df.grad(u), df.grad(v)) * dx
        + u * v * dx
        + gamma_u * df.inner(df.grad(u), df.grad(v)) * dx(Neumann)
        + boundary_penalty
        + gamma_div * u * v * dx(Neumann)
    )

    auz = gamma_u * df.inner(df.grad(u), z) * dx(Neumann) + gamma_div * u * df.div(
        z
    ) * dx(Neumann)
    auq = 0.0

    ayv = (
        df.inner(df.dot(y, n), v) * ds
        + gamma_u * df.inner(y, df.grad(v)) * dx(Neumann)
        + gamma_div * df.div(y) * v * dx(Neumann)
    )

    ayz = (
        gamma_u * df.inner(y, z) * dx(Neumann)
        + gamma_div * df.inner(df.div(y), df.div(z)) * dx(Neumann)
        + gamma_p
        * h ** (-2)
        * df.inner(df.dot(y, df.grad(phi)), df.dot(z, df.grad(phi)))
        * dx(Neumann)
    )
    ayq = gamma_p * h ** (-3) * df.inner(df.dot(y, df.grad(phi)), q * phi) * dx(Neumann)

    apv = 0.0
    apz = gamma_p * h ** (-3) * df.inner(p * phi, df.dot(z, df.grad(phi))) * dx(Neumann)
    apq = gamma_p * h ** (-4) * df.inner(p * phi, q * phi) * dx(Neumann)

    lv = df.inner(f, v) * dx + gamma_div * f * v * dx(Neumann)
    lz = df.inner(f, df.div(z)) * dx(Neumann) - gamma_p * h ** (-2) * df.inner(
        g * phi_abs, df.dot(z, df.grad(phi))
    ) * dx(Neumann)
    lq = -gamma_p * h ** (-3) * df.inner(g * phi_abs, q * phi) * dx(Neumann)

    a = [[auv, auz, auq], [ayv, ayz, ayq], [apv, apz, apq]]
    l = [lv, lz, lq]
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
    # FunctionSpace Pk
    V = df.FunctionSpace(mesh, "CG", degV)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    u_ex = df.Expression("sin(x[0])*exp(x[1])", degree=6, domain=mesh)
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    f = -df.div(df.grad(u_ex)) + u_ex
    g = (
        df.inner(df.grad(u_ex), df.grad(phi))
        / df.inner(df.grad(phi), df.grad(phi)) ** (0.5)
        + u_ex * phi
    )
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u), df.grad(v)) * df.dx + u * v * df.dx
    l = f * v * df.dx + g * v * df.ds
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(l)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble - start_assemble)
    start_standard = time.time()
    u = df.Function(V)
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
