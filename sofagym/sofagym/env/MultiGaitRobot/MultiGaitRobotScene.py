
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from MultiGaitRobotToolbox import rewardShaper, goalSetter

import sofagym.env.common.utils_rand as utils_rand
from splib.animation import AnimationManagerController
import os
import pdb

VISUALISATION = False

pathSceneFile = os.path.dirname(os.path.abspath(__file__))
pathMesh = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'
# Units: mm, kg, s.     Pressure in kPa = k (kg/(m.s^2)) = k (g/(mm.s^2) =  kg/(mm.s^2)

##########################################
# Reduced Basis Definition           #####
##########################################
modesRobot = pathSceneFile + "/ROM_data/modesQuadrupedWellConverged.txt"
nbModes = 63
modesPosition = [0 for i in range(nbModes)]

########################################################################
# Reduced Integration Domain for the PDMS membrane layer           #####
########################################################################
RIDMembraneFile = pathSceneFile + "/ROM_data/reducedIntegrationDomain_quadrupedMembraneWellConvergedNG005.txt"
weightsMembraneFile = pathSceneFile + "/ROM_data/weights_quadrupedMembraneWellConvergedNG005.txt"

#######################################################################
# Reduced Integration Domain for the main silicone body           #####
#######################################################################
RIDFile = pathSceneFile + '/ROM_data/reducedIntegrationDomain_quadrupedBodyWellConvergedNG003.txt'
weightsFile = pathSceneFile + '/ROM_data/weights_quadrupedBodyWellConvergedNG003.txt'

##############################################################
# Reduced Integration Domain in terms of nodes           #####
##############################################################
listActiveNodesFile = pathSceneFile + '/ROM_data/listActiveNodes_quadrupedBodyMembraneWellConvergedNG003and005.txt'

##########################################
# Reduced Order Booleans             #####
##########################################
# performECSWBoolBody = True
# performECSWBoolMembrane = True
# performECSWBoolMappedMatrix = True
# prepareECSWBool = False

def add_plugins(rootNode):
    rootNode.addObject('RequiredPlugin', name='SoftRobots', pluginName='SoftRobots')
    rootNode.addObject('RequiredPlugin', name='SofaPython', pluginName='SofaPython3')
    rootNode.addObject('RequiredPlugin', name='ModelOrderReduction', pluginName='ModelOrderReduction')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")
    rootNode.addObject('RequiredPlugin', name="SofaConstraint")
    rootNode.addObject('RequiredPlugin', name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    rootNode.addObject('RequiredPlugin', name="SofaLoader")
    rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
    rootNode.addObject('RequiredPlugin', name="SofaGeneralLoader")
    return rootNode

def add_visuals_and_solvers(rootNode, config, visu, simu, frictionCoeff=0.7):

    if visu:
        source = config["source"]
        target = config["target"]
        rootNode.addObject("DefaultVisualManagerLoop")
        rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideBoundingCollisionModels showForceFields '
                                                   'showInteractionForceFields hideWireframe')

        rootNode.addObject("LightManager")
        spotLoc = [0, 0, 1000]
        rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1.0])
        rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)
        rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
        #rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    
    if simu:
        rootNode.addObject('FreeMotionAnimationLoop')
        rootNode.addObject('GenericConstraintSolver', printLog=False, tolerance="1e-4", maxIterations="1000")
        rootNode.addObject('CollisionPipeline')
        rootNode.addObject('BruteForceDetection', name="N2")
        rootNode.addObject('CollisionResponse', response="FrictionContact", responseParams="mu="+str(frictionCoeff))
        rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance="2.5", contactDistance="0.5", angleCone="0.01")
        rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))
        rootNode.gravity.value = [0, 0, -9810]
    
    return rootNode


def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=pos) #position=[-10, 0.0, 0.0]
    goal_visu = goal.addChild('goalVisu')
    goal_visu.addObject('MeshOBJLoader', name='loader', filename=pathMesh+"ball.obj", scale3d=[5, 5, 5], translation = pos)
    #goal_visu.addObject('OglModel', src='@loader', color=[0, 1, 0, 1])
    #goal_visu.addObject('BarycentricMapping')
    return goal_mo, goal_visu

def createScene(rootNode, config={"source": [220, -500, 100],
                                  "target": [220, 0, 0],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    multigaitScale, frictionCoeff, centerPressure, legPressure  = utils_rand.set_initial_state_distr(config) # sample new random (or static) dynamics
    multigaitMass, PDMSPoissonRatio, PDMSYoungModulus,  EcoFlexPoissonRatio, EcoFlexYoungModulus = utils_rand.set_dynamic_params(config)
    
    if config['reduced']:
        performECSWBoolBody = True
        performECSWBoolMembrane = True
        performECSWBoolMappedMatrix = True
        prepareECSWBool = False
    else:
        performECSWBoolBody = False
        performECSWBoolMembrane = False
        performECSWBoolMappedMatrix = False
        prepareECSWBool = False

    pressure = rootNode.addChild("pressureConst")
    pressure.addObject('MechanicalObject', name="center", scale=centerPressure)
    pressure.addObject('MechanicalObject', name="leg", scale=legPressure)

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode = add_plugins(rootNode)
    rootNode = add_visuals_and_solvers(rootNode, config, visu, simu, frictionCoeff)

    rootNode.dt.value = 0.05

    solverNode = rootNode.addChild('solverNode')

    if simu:
        solverNode.addObject('EulerImplicit', name='odesolver',firstOrder="false", rayleighStiffness='0.01', rayleighMass='0.01', printLog=False)
        solverNode.addObject('SparseLDLSolver', name="preconditioner", template="CompressedRowSparseMatrixMat3x3d")
        solverNode.addObject('GenericConstraintCorrection', solverName='preconditioner')
        solverNode.addObject('MechanicalMatrixMapperMOR', template='Vec1d,Vec1d', object1='@./reducedModel/alpha', object2='@./reducedModel/alpha', nodeToParse='@./reducedModel/model', performECSW=performECSWBoolMappedMatrix, listActiveNodesPath=listActiveNodesFile,timeInvariantMapping1 = True,timeInvariantMapping2 = True, saveReducedMass=False, usePrecomputedMass=False, precomputedMassPath='ROM_data/quadrupedMass_reduced63modes.txt', fastMatrixProduct=False, printLog=False)


    ##########################################
    # FEM Reduced Model                      #
    ##########################################
    reducedModel = solverNode.addChild('reducedModel')
    reducedModel.addObject('MechanicalObject', template='Vec1d', name='alpha', position=modesPosition, printLog=False, showObjectScale=multigaitScale)
    ##########################################
    # FEM Model                              #
    ##########################################
    model = reducedModel.addChild('model')
    model.addObject('MeshVTKLoader', name='loader', filename=pathMesh+'full_quadriped_fine.vtk')
    model.addObject('TetrahedronSetTopologyContainer', src='@loader')
    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale=4e-5,
                    rx=0, printLog=False)
    model.addObject('ModelOrderReductionMapping', input='@../alpha', output='@./tetras', modesPath=modesRobot,
                    printLog=False, mapMatrices=0)
    model.addObject('UniformMass', name='quadrupedMass', totalMass=multigaitMass, printLog=False)
    model.addObject('HyperReducedTetrahedronFEMForceField', template='Vec3d',
                    name='Append_HyperReducedFF_QuadrupedWellConverged_'+str(nbModes)+'modes', method='large',
                    poissonRatio=EcoFlexPoissonRatio,  youngModulus=EcoFlexYoungModulus, prepareECSW=prepareECSWBool,
                    performECSW=performECSWBoolBody, nbModes=str(nbModes), modesPath=modesRobot, RIDPath=RIDFile,
                    weightsPath=weightsFile, nbTrainingSet=93, periodSaveGIE=50,printLog=False)
    model.addObject('BoxROI', name='boxROISubTopo', box=[0, 0, 0, 150, -100, 1], drawBoxes='true')
    model.addObject('BoxROI', name='membraneROISubTopo', box=[0, 0, -0.1, 150, -100, 0.1], computeTetrahedra=False,
                    drawBoxes=True)

    ##########################################
    # Sub topology                           #
    ##########################################
    modelSubTopo = model.addChild('modelSubTopo')
    modelSubTopo.addObject('TriangleSetTopologyContainer', position='@membraneROISubTopo.pointsInROI',
                           triangles="@membraneROISubTopo.trianglesInROI", name='container')
    modelSubTopo.addObject('HyperReducedTriangleFEMForceField', template='Vec3d', name='Append_subTopoFEM',
                           method='large', poissonRatio=PDMSPoissonRatio,  youngModulus=PDMSYoungModulus, prepareECSW=prepareECSWBool,
                           performECSW=performECSWBoolMembrane, nbModes=str(nbModes), modesPath=modesRobot,
                           RIDPath=RIDMembraneFile, weightsPath=weightsMembraneFile, nbTrainingSet=93,
                           periodSaveGIE=50, printLog=False)

    ##########################################
    # Constraint                             #
    ##########################################
    centerCavity = model.addChild('centerCavity')
    centerCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Center-cavity_finer.stl')
    centerCavity.addObject('MeshTopology', src='@loader', name='topo')
    centerCavity.addObject('MechanicalObject', name='centerCavity')
    centerCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                           value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                           valueType="volumeGrowth")
    centerCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    rearLeftCavity = model.addChild('rearLeftCavity')
    rearLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Left-cavity_finer.stl')
    rearLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearLeftCavity.addObject('MechanicalObject', name='rearLeftCavity')
    rearLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                             valueType="volumeGrowth", value=0.0000, triangles='@topo.triangles', drawPressure=0,
                             drawScale=0.0002)
    rearLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    rearRightCavity = model.addChild('rearRightCavity')
    rearRightCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Right-cavity_finer.stl')
    rearRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearRightCavity.addObject('MechanicalObject', name='rearRightCavity')
    rearRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                              value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                              valueType="volumeGrowth")
    rearRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    frontLeftCavity = model.addChild('frontLeftCavity')
    frontLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Front-Left-cavity_finer.stl')
    frontLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontLeftCavity.addObject('MechanicalObject', name='frontLeftCavity')
    frontLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                              value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                              valueType="volumeGrowth")
    frontLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    frontRightCavity = model.addChild('frontRightCavity')
    frontRightCavity.addObject('MeshSTLLoader', name='loader',
                               filename=pathMesh+'quadriped_Front-Right-cavity_finer.stl')
    frontRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontRightCavity.addObject('MechanicalObject', name='frontRightCavity')
    frontRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth")
    frontRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    if simu:
        modelCollis = model.addChild('modelCollis')
        modelCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_collision.stl',
                              rotation=[0, 0, 0], translation=[0, 0, 0], scale3d=multigaitScale)
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.addObject('TriangleCollisionModel', group=0)
        modelCollis.addObject('LineCollisionModel', group=0)
        modelCollis.addObject('PointCollisionModel', group=0)
        modelCollis.addObject('BarycentricMapping')

    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', name='loader', filename=pathMesh+"quadriped_collision.stl")
        modelVisu.addObject('OglModel', src='@loader', template='Vec3d', color=[0.7, 0.7, 0.7, 0.6], scale3d=multigaitScale)
        modelVisu.addObject('BarycentricMapping')

    planeNode = rootNode.addChild('Plane')
    planeNode.addObject('MeshOBJLoader', name='loader', filename="mesh/floorFlat.obj", triangulate="true")
    planeNode.addObject('MeshTopology', src="@loader")
    planeNode.addObject('MechanicalObject', src="@loader", rotation=[90, 0, 0], translation=[250, 35, -1], scale=15)

    if visu:
        planeNode.addObject('OglModel', name="Visual", src="@loader", color=[1, 1, 1, 0.5], rotation=[90, 0, 0],
                            translation=[250, 35, -1], scale=15)
    if simu:
        planeNode.addObject('TriangleCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('LineCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('PointCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('UncoupledConstraintCorrection')
        planeNode.addObject('EulerImplicitSolver', name='odesolver')
        planeNode.addObject('CGLinearSolver', name='Solver', iterations=500, tolerance=1e-5, threshold=1e-5)

    goal_mo, goal_visu = add_goal_node(rootNode, config['goalPos'])

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos']))
    rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalVisu=goal_visu, goalPos=config['goalPos']))

    
    if VISUALISATION:
        print(">> Add runSofa visualisation")
        from visualisation import ApplyAction, get_config
        # path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
        config = get_config("./config_a_la_main.txt")
        config_env = config['env']
        actions = config['actions']
        scale = config_env['scale_factor']

        rootNode.addObject(ApplyAction(name="ApplyAction", root=rootNode, actions=actions, scale=scale))

    return rootNode
