import Sofa.Core
import SofaRuntime

## Register all the common component in the factory.
SofaRuntime.importPlugin("SofaComponentAll")


def createScene(rootNode):
    rootNode.findData("gravity").value = [0.0, -9.81, 0.0]
    rootNode.findData("dt").value = 0.01

    confignode = rootNode.addChild("Config")
    confignode.addObject("RequiredPlugin", name="SofaMiscCollision", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaImplicitOdeSolver", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaBaseMechanics", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaLoader", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaMeshCollision", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaOpenglVisual", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaRigid", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaConstraint", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaSimpleFem", printLog=False)
    confignode.addObject("RequiredPlugin", name="SofaSparseSolver", printLog=False)

    confignode.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")

    # Collision function

    rootNode.addObject("DefaultPipeline")
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject(
        "GenericConstraintSolver",
        tolerance="1e-6",
        maxIterations="1000",
    )
    rootNode.addObject("BruteForceDetection")
    rootNode.addObject(
        "RuleBasedContactManager",
        responseParams="mu=" + str(0.0),
        name="Response",
        response="FrictionContact",
    )
    rootNode.addObject(
        "LocalMinDistance",
        alarmDistance=10,
        contactDistance=5,
        angleCone=0.01,
    )

    ### Mechanical model

    totalMass = 1.0
    volume = 1.0
    inertiaMatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # Creating the floor
    floor = rootNode.addChild("floor")

    floor.addObject(
        "MechanicalObject",
        name="mstate",
        template="Rigid3",
        translation2=[250.0, -300.0, 0.0],
        rotation2=[0.0, 0.0, 45.0],
        showObjectScale=5.0,
    )

    floor.addObject("UniformMass", name="mass")
    floorCollis = floor.addChild("collision")
    floorCollis.addObject(
        "MeshObjLoader",
        name="loader",
        filename="mesh/floor.obj",
        triangulate="true",
        scale=5.0,
    )
    floorCollis.addObject("MeshTopology", src="@loader")
    floorCollis.addObject("MechanicalObject")
    floorCollis.addObject("TriangleCollisionModel", moving=True, simulated=True)
    floorCollis.addObject("LineCollisionModel", moving=True, simulated=True)
    floorCollis.addObject("PointCollisionModel", moving=True, simulated=True)

    floorCollis.addObject("RigidMapping")

    #### visualization
    floorVisu = floor.addChild("VisualModel")
    floorVisu.loader = floorVisu.addObject(
        "MeshObjLoader", name="loader", filename="mesh/floor.obj"
    )
    floorVisu.addObject(
        "OglModel",
        name="model",
        src="@loader",
        scale3d=[5.0] * 3,
        color=[0.6, 0.7, 0.7],
        updateNormals=False,
    )
    floorVisu.addObject("RigidMapping")

    floor2 = rootNode.addChild("floor2")

    floor2.addObject(
        "MechanicalObject",
        name="mstate",
        template="Rigid3",
        translation2=[-250.0, -300.0, 0.0],
        rotation2=[0.0, 0.0, -45.0],
        showObjectScale=5.0,
    )

    floor2.addObject("UniformMass", name="mass")
    floor2Collis = floor2.addChild("collision")
    floor2Collis.addObject(
        "MeshObjLoader",
        name="loader",
        filename="mesh/floor.obj",
        triangulate="true",
        scale=5.0,
    )
    floor2Collis.addObject("MeshTopology", src="@loader")
    floor2Collis.addObject("MechanicalObject")
    floor2Collis.addObject("TriangleCollisionModel", moving=False, simulated=False)
    floor2Collis.addObject("LineCollisionModel", moving=False, simulated=False)
    floor2Collis.addObject("PointCollisionModel", moving=False, simulated=False)

    floor2Collis.addObject("RigidMapping")

    #### visualization
    floor2Visu = floor2.addChild("VisualModel")
    floor2Visu.loader = floor2Visu.addObject(
        "MeshObjLoader", name="loader", filename="mesh/floor.obj"
    )
    floor2Visu.addObject(
        "OglModel",
        name="model",
        src="@loader",
        scale3d=[5.0] * 3,
        color=[0.6, 0.7, 0.7],
        updateNormals=False,
    )
    floor2Visu.addObject("RigidMapping")

    # Creating the sphere

    sphere = rootNode.addChild("sphere")
    sphere.addObject("EulerImplicitSolver", name="cg_odesolver", printLog="false")
    sphere.addObject(
        "SparseLDLSolver",
        template="CompressedRowSparseMatrixd",
        name="sphere1_SparseLDLSolver",
        printLog="false",
    )
    sphere.addObject(
        "SparseGridRamificationTopology",
        name="grid",
        fileTopology="mesh/ball.obj",
    )
    sphere.addObject(
        "MechanicalObject",
        name="sphere1",
        scale3d=[5.0] * 3,
        translation="-250 500 0",
    )
    sphere.addObject(
        "UniformMass",
        name="mass",
        vertexMass=[totalMass, volume, inertiaMatrix[:]],
    )
    sphere.addObject(
        "HexahedronFEMForceField",
        name="FEM",
        youngModulus="500",
        poissonRatio="0.3",
    )
    sphere.addObject(
        "GenericConstraintCorrection",
        name="Torus1_ConstraintCorrection",
        printLog="0",
    )

    sphere.addObject(
        "MeshObjLoader",
        name="loader",
        filename="mesh/ball.obj",
        triangulate="true",
        scale=1.0,
    )

    sphere.addObject("TriangleCollisionModel")
    sphere.addObject("LineCollisionModel")
    sphere.addObject("PointCollisionModel")

    sphereVisu = sphere.addChild("VisualModel")
    sphereVisu.loader = sphereVisu.addObject(
        "MeshObjLoader", name="loader", filename="mesh/ball.obj"
    )
    sphereVisu.addObject(
        "OglModel",
        name="model",
        src="@loader",
        scale3d=[5.0] * 3,
        color=[1.0, 0.0, 1.0],
        updateNormals=False,
    )
    sphereVisu.addObject("BarycentricMapping")

    return rootNode
