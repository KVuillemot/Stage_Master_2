import dolfin as df
import matplotlib.pyplot as plt
import mshr
import time

# dolfin parameters
df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"

# degree of interpolation for V and Vphi
degV = 1
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
    Facet = df.MeshFunction(
        "size_t", mesh, mesh.topology().dim() - 1, 0
    )  # codimension 1
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)  # codimension 0

    for cell in df.cells(mesh):
        for facet in df.facets(cell):
            v1, v2, v3 = df.vertices(facet)
            if (
                phi(v1.point()) * phi(v2.point()) <= 0.0
                or phi(v1.point()) * phi(v3.point()) <= 0.0
                or phi(v2.point()) * phi(v3.point()) <= 0.0
                or df.near(phi(v1.point()) * phi(v2.point()), 0.0)
                or df.near(phi(v1.point()) * phi(v3.point()), 0.0)
                or df.near(phi(v2.point()) * phi(v3.point()), 0.0)
            ):  # si on est sur le bord
                Cell[cell] = 1
                for facett in df.facets(cell):
                    Facet[facett] = 1

    # Variationnal problem resolution
    V = df.FunctionSpace(mesh, "CG", degV)
    w_h = df.TrialFunction(V)
    v_h = df.TestFunction(V)
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0
    # Exact solution and computation of the right hand side term f
    u_ex = df.Expression(
        "exp(x[0])*sin(2*pi*x[1])*cos(2*pi*x[2])", degree=4, domain=mesh
    )  # solution exacte
    f = -df.div(df.grad(u_ex))
    u_D = u_ex * (1.0 + phi)

    # Modification of dolfin measures, to consider cells and facets
    dx = df.Measure("dx", mesh, subdomain_data=Cell)
    ds = df.Measure("ds", mesh)
    dS = df.Measure("dS", mesh, subdomain_data=Facet)

    if test_case == "sphere":
        sigma = 20.0  # Stabilization parameter
    elif test_case == "ellipsoid":
        sigma = 40.0
    # Creation of the bilinear and linear forms using stabilization terms and boundary condition
    a = (
        df.inner(df.grad(phi * w_h), df.grad(phi * v_h)) * dx
        - df.inner(df.grad(phi * w_h), n) * phi * v_h * ds
        + sigma
        * h_avg
        * df.jump(df.grad(phi * w_h), n)
        * df.jump(df.grad(phi * v_h), n)
        * dS(1)
        + sigma
        * h**2
        * df.div(df.grad(phi * w_h))
        * df.div(df.grad(phi * v_h))
        * dx(1)
    )
    L = (
        f * phi * v_h * dx
        - sigma * h**2 * f * df.div(df.grad(phi * v_h)) * dx(1)
        - sigma * h**2 * df.div(df.grad(phi * v_h)) * df.div(df.grad(u_D)) * dx(1)
        - df.inner(df.grad(u_D), df.grad(phi * v_h)) * dx
        + df.inner(df.grad(u_D), n) * phi * v_h * ds
        - sigma
        * h_avg
        * df.jump(df.grad(u_D), n)
        * df.jump(df.grad(phi * v_h), n)
        * dS(1)
    )

    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_phi.append(end_assemble - start_assemble)
    w_h = df.Function(V)
    start_solve = time.time()
    df.solve(A, w_h.vector(), B)
    end_solve = time.time()
    Time_solve_phi.append(end_solve - start_solve)
    Time_total_phi.append(Time_assemble_phi[-1] + Time_solve_phi[-1])
    u_h = phi * w_h + u_D
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
    H = 8 * 2 ** (
        i - 1
    )  # to have approximately the same precision as in the phi-fem computation
    mesh = mshr.generate_mesh(domain, H)
    print("Standard fem iteration : ", i)
    # FunctionSpace P1
    V = df.FunctionSpace(mesh, "CG", degV)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)
    u_ex = df.Expression(
        "exp(x[0])*sin(2*pi*x[1])*cos(2*pi*x[2])", degree=4, domain=mesh
    )  # solution exacte
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    u_D = u_ex * (1.0 + phi)
    f = -df.div(df.grad(u_ex))
    # Definition of the boundary to apply Dirichlet condition
    def boundary(x, on_boundary):
        return on_boundary

    bc = df.DirichletBC(V, u_D, boundary)
    # Resolution of the variationnal problem
    a = df.inner(df.grad(u), df.grad(v)) * df.dx
    L = f * v * df.dx
    start_assemble = time.time()
    A = df.assemble(a)
    B = df.assemble(L)
    end_assemble = time.time()
    Time_assemble_standard.append(end_assemble - start_assemble)
    bc.apply(A, B)  # apply Dirichlet boundary conditions to the problem
    start_standard = time.time()
    u = df.Function(V)
    df.solve(A, u.vector(), B)
    end_standard = time.time()
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

file_name = "poisson_dirichlet"
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
