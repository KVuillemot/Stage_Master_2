import dolfin as df
import vedo
from vedo import shapes
import vedo.dolfin as vdf
import mshr
import numpy as np
import matplotlib.pyplot as plt
from vedo.dolfin import plot, screenshot, show, Text2D, Plotter, shapes
from vedo import Latex
import sympy


plt.style.use("bmh")
params = {
    "axes.labelsize": 28,
    "font.size": 22,
    "axes.titlesize": 28,
    "legend.fontsize": 20,
    "figure.titlesize": 26,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.figsize": (10, 8),
    "legend.shadow": True,
    "patch.edgecolor": "black",
}
plt.rcParams.update(params)


case = "sphere_normals"
save_fig = False

bg_col = "gray5"
if case == "sphere_normals":
    mesh = vedo.Sphere(c="gray2").computeNormals().lineWidth(0).flat()

    vdf.show(mesh, bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/sphere_normals.png")

elif case == "mesh_standard_2D":
    mesh = mshr.Ellipse(df.Point(0, 0), 0.5, 1.0 / 3.0)
    mesh = mshr.generate_mesh(mesh, 20)
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/ellipse_std.png")

elif case == "mesh_standard_3D":
    mesh = mshr.Ellipsoid(df.Point(0, 0, 0), 0.5, 1.0 / 3.0, 1.0 / 3.0)
    mesh = mshr.generate_mesh(mesh, 30)
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/ellipsoid_std.png")

elif case == "shape_functions_1D":
    mesh = df.UnitIntervalMesh(1)
    for i in range(1, 3):
        V = df.FunctionSpace(mesh, "CG", i)
        u = df.Function(V)
        x = np.linspace(0, 1, 50)

        for j in range(V.dim()):
            u.vector()[:] = 0.0
            u.vector()[j] = 1.0

            y = np.fromiter(map(u, x), dtype=np.double)

            plt.plot(x, y, label=f"$j = {j}$")

        plt.xlim(0.0, 1.0)
        plt.xlabel("$x$")
        plt.ylabel(r"$\phi_j(x)$")
        plt.legend(loc="center left")
        if save_fig:
            plt.savefig(f"plots_rapport/shape_functions_1D_P{str(i)}.png")
        plt.show()
elif case == "shape_functions_2D":
    mesh = df.UnitSquareMesh(20, 20)
    V = df.FunctionSpace(mesh, "P", 1)
    v = df.Function(V)
    v.vector()[100] = 1
    vdf.plot(v, warpZfactor=1)
    if save_fig:
        vdf.screenshot(f"plots_rapport/shape_functions_2D_P1.png")

elif case == "nodes_triangle":

    x = np.array([0, 1, 0, 0])
    y = np.array([0, 0, 1, 0])
    plt.figure()

    plt.plot(x, y, color="black")

    X = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
    Y = ["black", "black", "black", "red", "red", "red"]
    labels = ["1", "2", "3", "4", "5", "6"]

    plt.scatter(X[:, 0], X[:, 1], s=170, color=Y[:])

    for i, txt in enumerate(labels):
        plt.annotate(txt, (X[i, 0] + 0.025, X[i, 1] + 0.025), color=Y[i])
    if save_fig:
        plt.savefig("plots_rapport/triangle.png")
    plt.show()

elif case == "nodes_quad":

    x = np.array([1, -1, -1, 1, 1])
    y = np.array([1, 1, -1, -1, 1])
    plt.figure()

    plt.plot(x, y, color="black")

    X = np.array(
        [
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
            [0.0, 1],
            [-1, 0],
            [0, -1],
            [1, 0.0],
            [0.0, 0.0],
        ]
    )
    Y = ["black", "black", "black", "black", "red", "red", "red", "red", "red"]
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    plt.scatter(X[:, 0], X[:, 1], s=170, color=Y[:])

    for i, txt in enumerate(labels):
        plt.annotate(txt, (X[i, 0] + 0.025, X[i, 1] + 0.025), color=Y[i])
    if save_fig:
        plt.savefig("plots_rapport/quad.png")
    plt.show()

elif case == "situation_phi_fem":
    background_mesh = df.RectangleMesh(df.Point(-0.7, -0.7), df.Point(0.7, 0.7), 16, 16)
    mesh = mshr.Ellipse(df.Point(0, 0), 0.5, 1.0 / 3.0)
    mesh = mshr.generate_mesh(mesh, 150)
    vdf.plot(background_mesh, c="gray", axes=0)
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0, lw=0)
    if save_fig:
        vdf.screenshot("plots_rapport/ellipse_phi_fem.png")

elif case == "selection_cells_phi_fem":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2

        def value_shape(self):
            return (2,)

    degPhi = 3
    background_mesh = df.RectangleMesh(df.Point(-0.7, -0.7), df.Point(0.7, 0.7), 16, 16)
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        v1, v2, v3 = df.vertices(cell)
        if (
            phi(v1.point()) <= 0.0
            or phi(v2.point()) <= 0.0
            or phi(v3.point()) <= 0.0
            or df.near(phi(v1.point()), 0.0)
            or df.near(phi(v2.point()), 0.0)
            or df.near(phi(v3.point()), 0.0)
        ):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)

    ellipse = mshr.Ellipse(df.Point(0, 0), 1.0 / 2.0, 1.0 / 3.0)
    ellipse = mshr.generate_mesh(ellipse, 150)
    vdf.plot(background_mesh, c="gray", axes=0)
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0)

    if save_fig:
        vdf.screenshot("plots_rapport/ellipse_phi_fem_cells.png")

elif case == "selection_boundary_cells_phi_fem":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2

        def value_shape(self):
            return (2,)

    degPhi = 2
    background_mesh = df.RectangleMesh(df.Point(-0.7, -0.7), df.Point(0.7, 0.7), 16, 16)
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        v1, v2, v3 = df.vertices(cell)
        if (
            phi(v1.point()) <= 0.0
            or phi(v2.point()) <= 0.0
            or phi(v3.point()) <= 0.0
            or df.near(phi(v1.point()), 0.0)
            or df.near(phi(v2.point()), 0.0)
            or df.near(phi(v3.point()), 0.0)
        ):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)
    # Selection of cells and facets on the boundary for Omega_h^Gamma
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    Facet.set_all(0)
    Cell.set_all(0)

    for cell in df.cells(mesh):
        for facet in df.facets(cell):
            v1, v2 = df.vertices(facet)
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                vc1, vc2, vc3 = df.vertices(cell)
                # Cells for dirichlet condition
                Cell[cell] = 1
                for facett in df.facets(cell):
                    Facet[facett] = 1
    boundary = df.SubMesh(mesh, Cell, 1)
    ellipse = mshr.Ellipse(df.Point(0, 0), 1.0 / 2.0, 1.0 / 3.0)
    ellipse = mshr.generate_mesh(ellipse, 150)

    vdf.plot(background_mesh, c="gray", axes=0, legend=r"$\mathcal{T}_h^{\mathcal{D}}$")
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0, legend=r"$\mathcal{T}_h$")
    vdf.plot(boundary, c="pink2", bg=bg_col, axes=0, legend=r"$\mathcal{T}_h^{\Gamma}$")
    if save_fig:
        vdf.screenshot("plots_rapport/ellipse_phi_fem_boundary_cells.png")


elif case == "selection_boundary_cells_mixed_phi_fem":

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0] * 2.0) ** 2 + (x[1] * 3.0) ** 2

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.0 - df.DOLFIN_EPS

    def neumann(point):
        return point.x() < 0.0 + df.DOLFIN_EPS

    def dirichlet_inter_neumann(point):
        return df.Near(point.x(), 0.0)

    degPhi = 2
    background_mesh = df.RectangleMesh(df.Point(-0.7, -0.7), df.Point(0.7, 0.7), 16, 16)
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        v1, v2, v3 = df.vertices(cell)
        if (
            phi(v1.point()) <= 0.0
            or phi(v2.point()) <= 0.0
            or phi(v3.point()) <= 0.0
            or df.near(phi(v1.point()), 0.0)
            or df.near(phi(v2.point()), 0.0)
            or df.near(phi(v3.point()), 0.0)
        ):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    # Selection of cells and facets on the boundary for Omega_h^Gamma
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_dirichlet_sub.set_all(0)
    facet_dirichlet_sub.set_all(0)
    vertices_dirichlet_sub.set_all(0)
    cell_neumann_sub.set_all(0)
    facet_neumann_sub.set_all(0)
    vertices_neumann_sub.set_all(0)

    Neumann, Dirichlet, Interface = 1, 2, 3
    for cell in df.cells(mesh):
        for facet in df.facets(cell):
            v1, v2 = df.vertices(facet)
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                Cell[cell] = Interface
                for facett in df.facets(cell):
                    Facet[facett] = Interface
                vc1, vc2, vc3 = df.vertices(cell)
                # Cells for dirichlet condition
                if (
                    dirichlet(vc1.point())
                    and dirichlet(vc2.point())
                    and dirichlet(vc3.point())
                ):
                    Cell[cell] = Dirichlet
                    cell_dirichlet_sub[cell] = 1
                    (
                        vertices_dirichlet_sub[vc1],
                        vertices_dirichlet_sub[vc2],
                        vertices_dirichlet_sub[vc3],
                    ) = (1, 1, 1)
                    for facett in df.facets(cell):
                        Facet[facett] = Dirichlet
                        facet_dirichlet_sub[facett] = 1

                # Cells for Neumann condition
                if (
                    neumann(vc1.point())
                    and neumann(vc2.point())
                    and neumann(vc3.point())
                ):
                    Cell[cell] = Neumann
                    cell_neumann_sub[cell] = 1
                    (
                        vertices_neumann_sub[vc1],
                        vertices_neumann_sub[vc2],
                        vertices_neumann_sub[vc3],
                    ) = (1, 1, 1)
                    for facett in df.facets(cell):
                        Facet[facett] = Neumann
                        facet_neumann_sub[facett] = 1

    ellipse = mshr.Ellipse(df.Point(0, 0), 1.0 / 2.0, 1.0 / 3.0)
    ellipse = mshr.generate_mesh(ellipse, 150)
    sub_dirichlet = df.SubMesh(mesh, Cell, Dirichlet)
    sub_neumann = df.SubMesh(mesh, Cell, Neumann)
    sub_interface = df.SubMesh(mesh, Cell, Interface)

    vdf.plot(background_mesh, c="gray", axes=0, legend=r"$\mathcal{T}_h^{\mathcal{D}}$")
    vdf.plot(mesh, c="cyan2", bg=bg_col, axes=0, legend=r"$\mathcal{T}_h$")
    vdf.plot(
        sub_dirichlet,
        c="blue2",
        bg=bg_col,
        axes=0,
        legend=r"$\mathcal{T}_h^{\Gamma_D}$",
    )
    vdf.plot(
        sub_neumann,
        c="purple1",
        bg=bg_col,
        axes=0,
        legend=r"$\mathcal{T}_h^{\Gamma_N}$",
    )
    vdf.plot(sub_interface, c="red", bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/ellipse_phi_fem_boundary_cells_mixed.png")

elif case == "liver_surface":
    mesh = vdf.Mesh("../Codes/Hyperelasticite/liver/data/liver1.stl")
    vdf.plot(mesh, c="cyan", bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/liver_surface.png")

elif case == "liver_volume":
    mesh = vdf.Mesh("../Codes/Hyperelasticite/liver/data/data_xml/liver1.xml")
    vdf.plot(mesh, c="cyan", bg=bg_col, axes=0)
    if save_fig:
        vdf.screenshot("plots_rapport/liver_volume_not_cutted.png")

elif case == "compute_f_non_linear":
    x, y = sympy.symbols("xx yy")
    u_ex = 1 + x + 2 * y

    def q(u):
        return 1 + u * u

    f = -sympy.diff(q(u_ex) * sympy.diff(u_ex, x), x) - sympy.diff(
        q(u_ex) * sympy.diff(u_ex, y), y
    )
    print(f)

elif case == "compute_f_linear":
    x, y = sympy.symbols("xx yy")
    u_ex = sympy.sin(x) * sympy.exp(y)

    f = -sympy.diff(sympy.diff(u_ex, x), x) - sympy.diff(sympy.diff(u_ex, y), y)
    print(f)

elif case == "plot_subdomains_mixed_conditions_facets":

    polV = 2
    degPhi = 2 + polV

    class phi_expr(df.UserExpression):
        def eval(self, value, x):
            value[0] = -1.0 + (x[0]) ** 2 + (x[1]) ** 2

        def value_shape(self):
            return (2,)

    def dirichlet(point):
        return point.x() > 0.0 - df.DOLFIN_EPS

    def neumann(point):
        return point.x() < 0.0 + df.DOLFIN_EPS

    def dirichlet_inter_neumann(point):
        return df.Near(point.x(), 0.0)

    degPhi = 2
    background_mesh = df.RectangleMesh(df.Point(-1.5, -1.5), df.Point(1.5, 1.5), 16, 16)
    V_phi = df.FunctionSpace(background_mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    Cell_omega = df.MeshFunction(
        "size_t", background_mesh, background_mesh.topology().dim()
    )
    Cell_omega.set_all(0)
    for cell in df.cells(background_mesh):
        v1, v2, v3 = df.vertices(cell)
        if (
            phi(v1.point()) <= 0.0
            or phi(v2.point()) <= 0.0
            or phi(v3.point()) <= 0.0
            or df.near(phi(v1.point()), 0.0)
            or df.near(phi(v2.point()), 0.0)
            or df.near(phi(v3.point()), 0.0)
        ):
            Cell_omega[cell] = 1
    mesh = df.SubMesh(background_mesh, Cell_omega, 1)

    # Creation of the FunctionSpace for Phi on Omega_h
    V_phi = df.FunctionSpace(mesh, "CG", degPhi)
    phi = phi_expr(element=V_phi.ufl_element())
    phi = df.interpolate(phi, V_phi)

    # Selection of cells and facets on the boundary for Omega_h^Gamma
    mesh.init(1, 2)
    Facet = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    Cell = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_dirichlet_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    cell_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim())
    facet_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    vertices_neumann_sub = df.MeshFunction("bool", mesh, mesh.topology().dim() - 2)
    Facet.set_all(0)
    Cell.set_all(0)
    cell_dirichlet_sub.set_all(0)
    facet_dirichlet_sub.set_all(0)
    vertices_dirichlet_sub.set_all(0)
    cell_neumann_sub.set_all(0)
    facet_neumann_sub.set_all(0)
    vertices_neumann_sub.set_all(0)

    Neumann, Dirichlet, Interface = 1, 2, 3
    for cell in df.cells(mesh):
        for facet in df.facets(cell):
            v1, v2 = df.vertices(facet)
            if phi(v1.point()) * phi(v2.point()) <= 0.0 or df.near(
                phi(v1.point()) * phi(v2.point()), 0.0
            ):
                # check if the cell is a cell for Dirichlet condition or Neumann condition and add every cells, facets, vertices to the restricition
                Cell[cell] = Interface
                for facett in df.facets(cell):
                    Facet[facett] = Interface
                vc1, vc2, vc3 = df.vertices(cell)
                # Cells for dirichlet condition
                if (
                    dirichlet(vc1.point())
                    and dirichlet(vc2.point())
                    and dirichlet(vc3.point())
                ):
                    Cell[cell] = Dirichlet
                    cell_dirichlet_sub[cell] = 1
                    (
                        vertices_dirichlet_sub[vc1],
                        vertices_dirichlet_sub[vc2],
                        vertices_dirichlet_sub[vc3],
                    ) = (1, 1, 1)
                    for facett in df.facets(cell):
                        Facet[facett] = Dirichlet
                        facet_dirichlet_sub[facett] = 1

                # Cells for Neumann condition
                if (
                    neumann(vc1.point())
                    and neumann(vc2.point())
                    and neumann(vc3.point())
                ):
                    Cell[cell] = Neumann
                    cell_neumann_sub[cell] = 1
                    (
                        vertices_neumann_sub[vc1],
                        vertices_neumann_sub[vc2],
                        vertices_neumann_sub[vc3],
                    ) = (1, 1, 1)
                    for facett in df.facets(cell):
                        Facet[facett] = Neumann
                        facet_neumann_sub[facett] = 1

    sub_dirichlet = df.SubMesh(mesh, Cell, Dirichlet)
    sub_neumann = df.SubMesh(mesh, Cell, Neumann)
    sub_interface = df.SubMesh(mesh, Cell, Interface)

    plt = vdf.plot(background_mesh, color="gray", lw=1.0, bg=bg_col, axes=0)
    # plt += plot(mesh, color="white", interactive = False, add=True)
    plt += vdf.plot(mesh, color="cyan2", add=True, lw=1.2, bg=bg_col, axes=0)

    plt += vdf.plot(
        sub_dirichlet, add=True, color="blue3", lw=3.0, bg=bg_col, alpha=1.0, axes=0
    )
    plt += vdf.plot(
        sub_neumann, add=True, color="purple3", lw=3.0, bg=bg_col, alpha=1.0, axes=0
    )

    om = vedo.shapes.Disc((0.0, 0.0, 0), 1, 1 + 0.01, c="black", res=60)
    plt += vdf.plot(om, add=True, alpha=1.0, axes=0)
    plt += vdf.plot(
        shapes.Arrow2D((0.9, 1.1, 0.0), (0.38, 0.82, 0), c="black"),
        add=True,
        bg=bg_col,
        axes=0,
    )  # dirichlet facets
    plt += vdf.plot(
        shapes.Arrow2D((0.9, 1.1, 0.0), (0.68, 0.56, 0), c="black"),
        add=True,
        bg=bg_col,
        axes=0,
    )  # dirichlet facets
    plt += vdf.plot(
        shapes.Arrow2D((-0.25, 0.3, 0.0), (-0.75, 0.44, 0), c="black"),
        add=True,
        bg=bg_col,
        axes=0,
    )  # neumann facets
    plt += vdf.plot(
        shapes.Arrow2D((-0.83, 1.1, 0.0), (-0.65, 0.75, 0), c="red"),
        add=True,
        bg=bg_col,
        axes=0,
    )  # not neumann facets
    plt += vdf.plot(
        shapes.Arrow2D((1.6, -0.20, 0.0), (1.0, 0, 0), c="black"),
        add=True,
        bg=bg_col,
        axes=0,
    )  # real boundary

    # plt += plot(df.RectangleMesh(df.Point(0.3,0.4), df.Point(0.5,0.5), 50,50), c='gray5',add=True, lw=3,axes=0)
    actors = plt.actors[0:9]
    gamma = r"\Gamma"
    formula_1 = Latex(gamma, c="k", s=0.8, usetex=False, res=60).pos(0.4, -1.35, 0)
    F_h_D = r"E \in \mathcal{F}_h^{\Gamma_D}"
    formula_2 = vedo.Latex(F_h_D, c="k", s=0.8, usetex=False, res=60).pos(
        -0.30, 0.25, 0
    )
    F_h_N = r"E \in \mathcal{F}_h^{\Gamma_N}"
    formula_3 = vedo.Latex(F_h_N, c="k", s=0.8, usetex=False, res=60).pos(
        -1.1, -0.85, 0
    )
    F_h_not_N = r"E \notin \mathcal{F}_h^{\Gamma_N}"
    formula_4 = vedo.Latex(F_h_not_N, c="r", s=0.8, usetex=False, res=60).pos(
        -2.0, 0.25, 0
    )
    vedo.show(actors, formula_1, formula_2, formula_3, formula_4, axes=0, bg=bg_col)
    if save_fig:
        vdf.screenshot("plots_rapport/plot_subdomains_mixed_conditions_facets.png")

elif case == "plot_displacement_hyperelasticity":

    domain = mshr.Circle(df.Point(0.5, 0.5), df.sqrt(2.0) / 4.0)
    mesh = mshr.generate_mesh(domain, 80)
    V = df.VectorFunctionSpace(mesh, "CG", dim=2, degree=2)
    u_ex = df.Expression(
        ("3.*cos(x[0])*sin(x[1])", "3.*sin(x[0])*cos(x[1])"), degree=3, domain=mesh
    )
    vdf.plot(df.project(u_ex, V))


elif case == "plot_force_hyperelasticity":

    x, y = sympy.symbols("xx yy")
    u_ex = 3.0 * sympy.cos(x) * sympy.sin(y)

    f = -sympy.diff(sympy.diff(u_ex, x), x) - sympy.diff(sympy.diff(u_ex, y), y)
    print(f)
    domain = mshr.Circle(df.Point(0.5, 0.5), df.sqrt(2.0) / 4.0)
    mesh = mshr.generate_mesh(domain, 80)
    V = df.VectorFunctionSpace(mesh, "CG", dim=2, degree=2)
    f = df.Expression(
        ("6.0*sin(x[1])*cos(x[0])", "6.0*sin(x[1])*cos(x[0])"), degree=3, domain=mesh
    )
    vdf.plot(df.project(f, V))
