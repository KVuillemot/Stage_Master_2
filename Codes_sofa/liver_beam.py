import sys, os
import meshio
from pathlib import Path
import numpy as np

sys.path.append(os.path.abspath("./bindings/Sofa/package"))
sys.path.append(os.path.abspath("./bindings/SofaRuntime/package"))
sys.path.append(os.path.abspath("./bindings/SofaTypes/package"))

import Sofa


def createScene(root):
    root.addObject("RequiredPlugin", name="SofaComponentAll")
    root.addObject("RequiredPlugin", name="SofaOpenglVisual")
    root.addObject("RequiredPlugin", name="SofaSparseSolver")
    root.addObject("RequiredPlugin", name="SofaMiscCollision", printLog=False)
    root.addObject("RequiredPlugin", name="SofaConstraint")
    root.addObject("RequiredPlugin", name="SofaImplicitOdeSolver")
    root.addObject("RequiredPlugin", name="SofaLoader")
    root.addObject("RequiredPlugin", name="SofaMeshCollision")
    root.addObject("RequiredPlugin", name="SofaRigid")
    root.addObject("RequiredPlugin", name="SofaTopologyMapping")
    root.addObject("RequiredPlugin", name="SofaBoundaryCondition")
    root.addObject("RequiredPlugin", name="SofaEngine")
    root.addObject("RequiredPlugin", name="SofaBaseMechanics")
    root.addObject("RequiredPlugin", name="SofaGeneralVisual")
    root.addObject("RequiredPlugin", name="SofaSimpleFem")

    root.addObject(
        "VisualStyle",
        displayFlags="showVisualModels showBehaviorModels  showForceFields",
    )
    root.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")

    root.dt = 0.02
    root.gravity = [0.0, -9.81, 0.0]
    root.addObject("DefaultPipeline", verbose="0", draw="0")
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject(
        "MinProximityIntersection",
        name="Proximity",
        alarmDistance=".8",
        contactDistance=".5",
    )
    root.addObject(
        "DefaultContactManager", name="collision response", response="default"
    )
    ### Mechanical model
    dx_floor = "-35."
    dy_floor = "-50."
    dz_floor = "0."

    # Creating the floor
    floor = root.addChild("Floor")
    floor.addObject("MeshObjLoader", name="loader", filename="mesh/floor.obj")
    floor.addObject("MeshTopology", src="@loader")
    floor.addObject(
        "MechanicalObject",
        src="@loader",
        dy=dy_floor,
        dx=dx_floor,
        dz=dz_floor,
        rotation="0 0 0",
        scale="1",
    )
    floor.addObject("TriangleCollisionModel", moving=False, simulated=False)
    floor.addObject("LineCollisionModel", moving=False, simulated=False)
    floor.addObject("PointCollisionModel", moving=False, simulated=False)

    # visual model for the floor
    floor.addObject(
        "MeshObjLoader",
        name="meshLoader_1",
        filename="mesh/floor.obj",
        scale="1.",
        handleSeams="1",
    )
    floor.addObject(
        "OglModel",
        name="FloorV",
        src="@meshLoader_1",
        color=[0.6, 0.7, 0.7],
        dx=dx_floor,
        dy=dy_floor,
        dz=dz_floor,
        rotation="0 0 0",
    )

    # Creating the wall
    rotation_wall = "90 0 0"
    dx_wall, dy_wall, dz_wall = dx_floor, "20", "-50"
    wall = root.addChild("wall")
    wall.addObject(
        "MeshObjLoader", name="loader", filename="mesh/floor.obj", scale="1."
    )
    wall.addObject("MeshTopology", src="@loader")
    wall.addObject(
        "MechanicalObject",
        src="@loader",
        dy=dy_wall,
        dx=dx_wall,
        dz=dz_wall,
        rotation=rotation_wall,
        scale="1",
    )
    wall.addObject("TriangleCollisionModel", moving=False, simulated=False)
    wall.addObject("LineCollisionModel", moving=False, simulated=False)
    wall.addObject("PointCollisionModel", moving=False, simulated=False)

    # visual model for the wall
    wall.addObject(
        "MeshObjLoader",
        name="meshLoader_1",
        filename="mesh/floor.obj",
        scale="1.",
        handleSeams="1",
    )
    wall.addObject(
        "OglModel",
        name="WallV",
        src="@meshLoader_1",
        color=[0.6, 0.7, 0.7],
        dx=dx_wall,
        dy=dy_wall,
        dz=dz_wall,
        rotation=rotation_wall,
    )

    floor2 = root.addChild("Floor2")
    floor2.addObject("MeshObjLoader", name="loader", filename="mesh/floor3.obj")
    floor2.addObject("MeshTopology", src="@loader")
    floor2.addObject(
        "MechanicalObject",
        src="@loader",
        dy=dy_floor,
        dx="-10",
        dz=dz_floor,
        rotation="0 0 0",
        scale="1",
    )
    floor2.addObject("TriangleCollisionModel", moving=True, simulated=False)
    floor2.addObject("LineCollisionModel", moving=True, simulated=False)
    floor2.addObject("PointCollisionModel", moving=True, simulated=False)

    # visual model for the floor
    floor2.addObject(
        "MeshObjLoader",
        name="meshLoader_1",
        filename="mesh/floor3.obj",
        scale="1.",
        handleSeams="1",
    )
    floor2.addObject(
        "OglModel",
        name="FloorV",
        src="@meshLoader_1",
        texturename="textures/brushed_metal.bmp",
        dx="-10",
        dy=dy_floor,
        dz=dz_floor,
        rotation="0 0 0",
    )

    # add an elastic beam (uses caribou)
    root.addObject(
        "RegularGridTopology",
        name="grid",
        min=[-31.5, -2.5, -55],
        max=[0.5, 7.5, -10],
        n=[3, 3, 9],
    )
    beam = root.addChild("beam")
    beam.addObject(
        "BackwardEulerODESolver",
        newton_iterations=10,
        rayleigh_stiffness=0,
        rayleigh_mass=0,
        residual_tolerance_threshold=1e-5,
        pattern_analysis_strategy="ALWAYS",
        printLog=True,
    )
    beam.addObject("LDLTSolver", backend="Eigen")
    beam.addObject("MechanicalObject", name="mo", src="@../grid")
    beam.addObject("TriangleCollisionModel", moving=True, simulated=False)
    beam.addObject("LineCollisionModel", moving=True, simulated=False)
    beam.addObject("PointCollisionModel", moving=True, simulated=False)
    # Complete hexa container
    beam.addObject(
        "HexahedronSetTopologyContainer", src="@../grid", name="mechanical_topology"
    )

    # - Mechanics
    beam.addObject(
        "SaintVenantKirchhoffMaterial", young_modulus=15000, poisson_ratio=0.3
    )
    beam.addObject("HyperelasticForcefield")

    # - Mass
    beam.addObject("HexahedronSetGeometryAlgorithms")
    beam.addObject("DiagonalMass", massDensity=0.2)

    # Fix the left side of the beam
    beam.addObject(
        "BoxROI",
        name="fixed_roi",
        quad="@surface_topology.quad",
        box=[-31.5, -2.5, -55.5, 0.5, 7.5, -54.9],
    )
    beam.addObject("FixedConstraint", indices="@fixed_roi.indices")

    # add the liver
    liver = root.addChild("liver")
    liver.addObject(
        "EulerImplicitSolver",
        name="cg_odesolver",
        printLog=False,
        rayleighStiffness="0.1",
        rayleighMass="0.1",
    )
    liver.addObject("CGLinearSolver", name="linear_solver")

    liver.addObject(
        "MeshGmshLoader", name="loader", filename="mesh/liver.msh", scale="3"
    )
    liver.addObject(
        "MechanicalObject",
        name="StateVectors",
        src="@loader",
        dx="-20",
        dy="100",
        dz="-10",
        rotation="0 -90 0",
    )
    liver.addObject(
        "TetrahedronSetTopologyContainer", name="TetraTopologyContainer", src="@loader"
    )
    liver.addObject("UniformMass", totalMass="6")
    liver.addObject(
        "TetrahedronFEMForceField",
        name="FEM",
        youngModulus="5000",
        poissonRatio="0.4",
        computeGlobalMatrix="false",
        updateStiffnessMatrix="false",
        method="large",
    )

    surf = liver.addChild("Surf")
    surf.addObject("MeshObjLoader", name="loader", filename="mesh/liver.obj", scale="3")
    surf.addObject("MeshTopology", src="@loader")
    surf.addObject(
        "MechanicalObject",
        src="@loader",
        dx="-20",
        dy="100",
        dz="-10",
        rotation="0 -90 0",
    )
    surf.addObject("TriangleCollisionModel")
    surf.addObject("LineCollisionModel")
    surf.addObject("PointCollisionModel")
    surf.addObject("SubsetMapping")

    livervisu = surf.addChild("visu")
    livervisu.addObject(
        "MeshObjLoader",
        name="loader",
        filename="mesh/liver.obj",
        scale="3.",
        handleSeams="1",
    )
    livervisu.addObject(
        "OglModel",
        name="liver",
        src="@loader",
        color=[1, 0, 1],
        dx="-20",
        dy="100",
        dz="-10",
        rotation="0 -90 0",
    )
    livervisu.addObject("BarycentricMapping")
    return root
