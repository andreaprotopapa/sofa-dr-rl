import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import SofaRuntime
from splib.animation import AnimationManagerController
from math import cos, sin
import numpy as np
import random
from splib.objectmodel import SofaPrefab, SofaObject
from splib.numerics import Vec3, Quat
from Cube import Cube
import sofagym.env.common.utils_rand as utils_rand

import pdb

from TrunkCubeToolbox import rewardShaper, goalSetter

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'


SofaRuntime.importPlugin("SofaComponentAll")

def add_plugins(rootNode):
    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject("RequiredPlugin", name="SofaPython3")
    rootNode.addObject('RequiredPlugin', name='BeamAdapter')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaMiscCollision")
    rootNode.addObject("RequiredPlugin", name="SofaBoundaryCondition")
    rootNode.addObject("RequiredPlugin", name="SofaConstraint")
    rootNode.addObject('RequiredPlugin', name='SofaImplicitOdeSolver')
    rootNode.addObject('RequiredPlugin', name='SofaLoader')
    rootNode.addObject('RequiredPlugin', name='SofaGeneralLoader')
    rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")

    rootNode.addObject("RequiredPlugin", name="SofaEngine") 
    rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
    rootNode.addObject('RequiredPlugin', name="SofaMiscFem")
    rootNode.addObject('RequiredPlugin', name="SofaSimpleFem")
    rootNode.addObject('RequiredPlugin', name="SofaRigid")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Algorithm")
    
    return rootNode

def add_visuals_and_solvers(root, config, visu, simu, fricionCoeff):
    if visu:
        source = config["source"]
        target = config["target"]
        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideMappings hideForceFields hideWireframe')
                                                       
        # rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields showInteractionForceFields hideWireframe')
        

        root.addObject("LightManager")

        spotLoc = [2*source[0], 0, 0]
        root.addObject("SpotLight", position=spotLoc, direction=[-np.sign(source[0]), 0.0, 0.0])
        root.addObject("InteractiveCamera", name='camera', position=source, lookAt=target, zFar=500)
        #root.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    if simu:
        root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
        root.addObject('BruteForceDetection')
        root.addObject('RuleBasedContactManager', responseParams="mu="+str(fricionCoeff), name='Response',
                           response='FrictionContactConstraint')
        root.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.2)

        root.addObject(AnimationManagerController(name="AnimationManager"))

        root.gravity.value = [0., -9810, 0.]
    return root

def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=pos) #position=[0.0, 0.0, 0.0] or [0.0, -100.0, 100.0]
    goal_visu = goal.addChild('goalVisu')
    goal_visu.addObject('MeshOBJLoader', name='loader', filename=path+"ball.obj", scale3d=[5, 5, 5], translation = pos)
    goal_visu.addObject('OglModel', src='@loader', color=[0, 1, 0, 1])
    goal_visu.addObject('BarycentricMapping')
    return goal_mo, goal_visu

def add_box_point(root, name, pos, color):
    box_point = root.addChild("Box_"+name)
    box_point.addObject('VisualStyle', displayFlags="showCollisionModels")
    box_point.addObject('MechanicalObject', name='mbox', showObject=False, drawMode="1", showObjectScale=2,
                             showColor=color, position=pos)
    box_visu = box_point.addChild('boxVisu')
    box_visu.addObject('MeshOBJLoader', name='loader', filename=path+"ball.obj", scale3d=[2, 2, 2], translation = pos)
    box_visu.addObject('OglModel', src='@loader', color=color)
    box_visu.addObject('BarycentricMapping')

def add_box(root, name, pos_init, pos_low, pos_high, color):
    pos_0 = pos_low
    pos_1 = [pos_low[0], 0.0, pos_high[2]]
    pos_2 = [pos_high[0], 0.0, pos_low[2]]
    pos_3 = pos_high
    add_box_point(root, name+"_0", list(np.array(pos_0) + np.array(pos_init)), color)
    add_box_point(root, name+"_1", list(np.array(pos_1) + np.array(pos_init)), color)
    add_box_point(root, name+"_2", list(np.array(pos_2) + np.array(pos_init)), color)
    add_box_point(root, name+"_3", list(np.array(pos_3) + np.array(pos_init)), color)
    


def effectorTarget(parentNode, position=[0., 0., 200]):
    target = parentNode.addChild("Target")
    target.addObject("EulerImplicitSolver", firstOrder=True)
    target.addObject("CGLinearSolver")
    target.addObject("MechanicalObject", name="dofs", position=position, showObject=True, showObjectScale=3,
                     drawMode=2, showColor=[1., 1., 1., 1.])
    target.addObject("UncoupledConstraintCorrection")
    return target

@SofaPrefab
class Trunk(SofaObject):
    """ This prefab is implementing a soft robot inspired by the elephant's trunk.
        The robot is entirely soft and actuated with 8 cables.
        The prefab is composed of:
        - a visual model
        - a collision model
        - a mechanical model for the deformable structure
        The prefab has the following parameters:
        - youngModulus
        - poissonRatio
        - totalMass
    """

    def __init__(self, parentNode, youngModulus=4500, poissonRatio=0.45, totalMass=0.42, scale = [1 , 1 , 1], inverseMode=False):
        self.scale = scale
        self.inverseMode = inverseMode
        self.node = parentNode.addChild('Trunk')

        self.node.addObject('MeshVTKLoader', name='loader', filename=path+'trunk.vtk')
        self.node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.node.addObject('TetrahedronSetTopologyModifier')
        self.node.addObject('TetrahedronSetGeometryAlgorithms')

        self.node.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false',
                            showIndicesScale='4e-5', showObjectScale=self.scale)
        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                            poissonRatio=poissonRatio,  youngModulus=youngModulus)

        self.__addCables()

    def __addCables(self):
        length1 = 10.
        length2 = 2.
        lengthTrunk = 195.

        pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
        direction = Vec3(0., length2-length1, lengthTrunk)
        direction.normalize()

        nbCables = 4

        self.cables = []
        for i in range(0, nbCables):
            theta = 1.57*i
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*20
            for k in range(0, 20, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableL = self.node.addChild('cableL'+str(i))
            cableL.addObject('MechanicalObject', name='meca', showObject=False, showColor=[1, 1, 0, 1], position=pullPoint[i]+[pos.toList() for pos in position])

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            cableL.addObject('CableConstraint' if not self.inverseMode else 'CableActuator', template='Vec3d',
                             name="cable", hasPullPoint="0", indices=idx, maxPositiveDisp='70', maxDispVariation="1",
                             minForce=0)
            cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
            self.cables.append(cableL)

        for i in range(0, nbCables):
            theta = 1.57*i
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*10
            for k in range(0, 9, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableS = self.node.addChild('cableS'+str(i))
            cableS.addObject('MechanicalObject', name='meca', position=pullPoint[i]+[pos.toList() for pos in position])

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            cableS.addObject('CableConstraint' if not self.inverseMode else 'CableActuator', template='Vec3d',
                             name="cable", hasPullPoint="0", indices=idx, maxPositiveDisp='40', maxDispVariation="1",
                             minForce=0)
            cableS.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')
            self.cables.append(cableS)

    def addVisualModel(self, color=[1., 1., 1., 1.]):
        trunkVisu = self.node.addChild('VisualModel')
        trunkVisu.addObject('MeshSTLLoader', filename=path+"trunk.stl")
        trunkVisu.addObject('OglModel', template='Vec3d', color=color, scale3d=self.scale)
        trunkVisu.addObject('BarycentricMapping')

    def addCollisionModel(self, selfCollision=False):
        #self.node.addObject('LinearSolverConstraintCorrection', name='GCS', solverName='precond')
        trunkColli = self.node.addChild('CollisionModel')
        for i in range(2):
            part = trunkColli.addChild("Part"+str(i+1))
            part.addObject('MeshSTLLoader', name="loader", filename=path+"trunk_colli"+str(i+1)+".stl", scale3d=self.scale)
            part.addObject('MeshTopology', src="@loader")
            part.addObject('MechanicalObject')
            part.addObject('PointCollisionModel', group=1 if not selfCollision else i)
            part.addObject('LineCollisionModel', group=1 if not selfCollision else i)
            part.addObject('TriangleCollisionModel', group=1 if not selfCollision else i)
            part.addObject('BarycentricMapping')

    def fixExtremity(self):
        self.node.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.node.addObject('PartialFixedConstraint', fixedDirections="1 1 1", indices="@boxROI.indices")

    def addEffectors(self, position=[0., 0., 195.]):
        effectors = self.node.addChild("Effectors")
        effectors.addObject("MechanicalObject", position=position)
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

def CreateObject(node, name, surfaceMeshFileName, visu, simu, translation=[0., 0., 0.], rotation=[0., 0., 0.],
                 uniformScale=1., totalMass=1.,
                 color=[1., 1., 0.], isAStaticObject=False):

    object = node.addChild(name)

    object.addObject('MechanicalObject', name="mstate", template="Rigid3", translation2=translation,
                     rotation2=rotation, showObjectScale=uniformScale)

    object.addObject('UniformMass', name="mass", totalMass=totalMass)

    # if not isAStaticObject:
    #     object.addObject('UncoupledConstraintCorrection')
    #     object.addObject('EulerImplicitSolver', name='odesolver')
    #     object.addObject('CGLinearSolver', name='Solver')

    # collision
    if simu:
        objectCollis = object.addChild('collision')
        objectCollis.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName, triangulate="true",
                               scale=uniformScale)

        objectCollis.addObject('MeshTopology', src="@loader")
        objectCollis.addObject('MechanicalObject')

        movement = not isAStaticObject
        objectCollis.addObject('TriangleCollisionModel', moving=movement, simulated=movement)
        objectCollis.addObject('LineCollisionModel', moving=movement, simulated=movement)
        objectCollis.addObject('PointCollisionModel', moving=movement, simulated=movement)

        objectCollis.addObject('RigidMapping')


    # visualization
    if visu:
        objectVisu = object.addChild("VisualModel")

        objectVisu.loader = objectVisu.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName)

        objectVisu.addObject('OglModel', name="model", src="@loader", scale3d=[uniformScale]*3, color=color,
                             updateNormals=False)

        objectVisu.addObject('RigidMapping')

    return object



def createScene(rootNode, config={"source": [-600.0, -25, 100],
                                  "target": [30, -25, 100],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    if config["unmodeled"]:
        cube_trasl, cube_rot, cubeScale, trunkScale, trunkYoungModulus = utils_rand.set_initial_state_distr(config) # sample new random (or static) dynamics
        cube_mass, frictionCoeff, trunkMass, trunkPoissonRatio= utils_rand.set_dynamic_params(config)
    else:
        cube_trasl, cube_rot, cubeScale, trunkScale = utils_rand.set_initial_state_distr(config) # sample new random (or static) dynamics
        cube_mass, frictionCoeff, trunkMass, trunkPoissonRatio, trunkYoungModulus = utils_rand.set_dynamic_params(config)

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode = add_plugins(rootNode)

    rootNode = add_visuals_and_solvers(rootNode, config, visu, simu, frictionCoeff)

    rootNode.dt.value = 0.01

    CreateObject(rootNode, name="Floor", surfaceMeshFileName="mesh/floor.obj", visu=visu, simu=simu, color=[1, 0.5, 0.5, 0.5],
                  uniformScale=4, rotation=[0, 0, 0], translation=[0.0, config["floorHeight"], 0.0], isAStaticObject=True)

    simulation = rootNode.addChild("Simulation")
    if simu:
        simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder="0", rayleighMass="0.1",
                             rayleighStiffness="0.1")
        simulation.addObject('SparseLDLSolver', name='precond')
        simulation.addObject('GenericConstraintCorrection', solverName="precond")
        
    trunk = Trunk(simulation, youngModulus=trunkYoungModulus, poissonRatio=trunkPoissonRatio, totalMass=trunkMass, scale=trunkScale, inverseMode=False)
    rootNode.trunk = trunk

    # vizualize cable trackers
    # colors = [[1, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1]]
    # for i, cable in enumerate(rootNode.trunk.cables[:4]):
    #     for j in range(len(cable.meca.position)):
    #         add_box_point(rootNode, f"cable_{i}_{j}", cable.meca.position[j], colors[i])

    if simu:
        trunk.addCollisionModel()
    if visu:
        trunk.addVisualModel(color=[1., 1., 1., 0.8])
    trunk.fixExtremity()
    trunk.addEffectors()

    cubeNode = CreateObject(simulation, name="Cube", surfaceMeshFileName="mesh/smCube27.obj", visu=visu, simu=simu, color=[1., 1., 0.],
                  isAStaticObject=False,
                  translation=cube_trasl, 
                  rotation=cube_rot,  
                  totalMass=cube_mass, 
                  uniformScale=cubeScale)
    
    goal_mo, goal_visu = add_goal_node(rootNode, config['goalPos'])

    if config["cubeTranslation_distr"] == "uniform":
        pos_init = config["cubeTranslation_init"]
        pos_init[1] = config["floorHeight"]
        add_box(rootNode, "cube", pos_init, config["cubeTranslation_unifLow"], config["cubeTranslation_unifHigh"], [0, 0, 1, 1])
    
    if config["goalList"] == None:
        pos_init = [0.0, 0.0, 0.0]
        pos_init[1] = config["floorHeight"]
        pos_low = [config["goal_low"][0], 0.0, config["goal_low"][1]]
        pos_high = [config["goal_high"][0], 0.0, config["goal_high"][1]]
        add_box(rootNode, "goal", pos_init, pos_low, pos_high, [1, 0, 1, 1])

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos'], effMO=cubeNode.mstate))
    rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalVisu=goal_visu, goalPos=config['goalPos']))
    return rootNode
