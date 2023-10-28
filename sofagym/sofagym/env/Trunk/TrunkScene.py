
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from splib.animation import AnimationManagerController
from math import cos, sin
import numpy as np
from splib.objectmodel import SofaPrefab, SofaObject
from splib.numerics import Vec3, Quat

import sofagym.env.common.utils_rand as utils_rand

from TrunkToolbox import rewardShaper, goalSetter

import os
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

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
    rootNode.addObject("RequiredPlugin", name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name='SofaImplicitOdeSolver')
    rootNode.addObject('RequiredPlugin', name='SofaLoader')
    rootNode.addObject('RequiredPlugin', name="SofaSimpleFem")
    return rootNode

def add_visuals_and_solvers(root, config, visu, simu, fricionCoeff=0.3):
    if visu:
        source = config["source"]
        target = config["target"]
        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideMappings hideForceFields hideWireframe')
        root.addObject("LightManager")

        spotLoc = [0, 0, 2*source[2]]
        root.addObject("SpotLight", position=spotLoc, direction=[0.0, 0.0, -np.sign(source[2])])
        root.addObject("InteractiveCamera", name='camera', position=source, lookAt=target, zFar=500)
        root.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    if simu:
        root.addObject('DefaultPipeline')
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")
        root.addObject('BruteForceDetection')
        root.addObject('RuleBasedContactManager', responseParams="mu="+str(fricionCoeff), name='Response',
                           response='FrictionContactConstraint')
        root.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.01)

        root.addObject(AnimationManagerController(name="AnimationManager"))

        root.gravity.value = [0., -9810., 0.]

    return root

def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=pos) #position=[0.0, 0.0, 0.0] or [0.0, -100.0, 100.0]
    goal_visu = goal.addChild('goalVisu')
    goal_visu.addObject('MeshOBJLoader', name='loader', filename=path+"ball.obj", scale3d=[3, 3, 3], translation = pos)
    goal_visu.addObject('OglModel', src='@loader', color=[1, 0, 0, 1])
    goal_visu.addObject('BarycentricMapping')
    return goal_mo, goal_visu


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

def add_box_point(root, name, pos, color):
    box_point = root.addChild("Box_"+name)
    box_point.addObject('VisualStyle', displayFlags="showCollisionModels")
    box_point.addObject('MechanicalObject', name='mbox', showObject=False, drawMode="1", showObjectScale=2,
                             showColor=color, position=pos)
    box_visu = box_point.addChild('boxVisu')
    box_visu.addObject('MeshOBJLoader', name='loader', filename=path+"ball.obj", scale3d=[2, 2, 2], translation = pos)
    box_visu.addObject('OglModel', src='@loader', color=color)
    box_visu.addObject('BarycentricMapping')


def add_box(root, name, pos_low, pos_high, color):
    pos_0 = pos_low
    pos_1 = [pos_low[0],  pos_low[1],  pos_high[2]]
    pos_2 = [pos_high[0], pos_low[1],  pos_low[2]]
    pos_3 = [pos_low[0],  pos_high[1], pos_low[2]]
    pos_4 = [pos_high[0], pos_high[1], pos_low[2]]
    pos_5 = [pos_low[0],  pos_high[1], pos_high[2]]
    pos_6 = [pos_high[0], pos_low[1],  pos_high[2]]
    pos_7 = pos_high
    add_box_point(root, name+"_0", list(np.array(pos_0)), color)
    add_box_point(root, name+"_1", list(np.array(pos_1)), color)
    add_box_point(root, name+"_2", list(np.array(pos_2)), color)
    add_box_point(root, name+"_3", list(np.array(pos_3)), color)
    add_box_point(root, name+"_4", list(np.array(pos_4)), color)
    add_box_point(root, name+"_5", list(np.array(pos_5)), color)
    add_box_point(root, name+"_6", list(np.array(pos_6)), color)
    add_box_point(root, name+"_7", list(np.array(pos_7)), color)


def createScene(rootNode, config={"source": [-600.0, -25, 100],
                                  "target": [30, -25, 100],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    trunkScale = utils_rand.set_initial_state_distr(config) # sample new random (or static) dynamics
    trunkMass, trunkPoissonRatio, trunkYoungModulus = utils_rand.set_dynamic_params(config)

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode = add_plugins(rootNode)

    rootNode = add_visuals_and_solvers(rootNode, config, visu, simu)
    
    rootNode.dt.value = 0.01

    simulation = rootNode.addChild("Simulation")

    if simu:
        simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder="0", rayleighMass="0.1",
                             rayleighStiffness="0.1")
        #simulation.addObject('ShewchukPCGLinearSolver', name='linearSolver', iterations='500', tolerance='1.0e-18',
        #                     preconditioners="precond")
        simulation.addObject('SparseLDLSolver', name='precond')
        #rootNode.addObject('SparseLDLSolver', name="preconditioner", template="CompressedRowSparseMatrixd")
        simulation.addObject('GenericConstraintCorrection', solverName="precond")

    trunk = Trunk(simulation, youngModulus=trunkYoungModulus, poissonRatio=trunkPoissonRatio, totalMass=trunkMass, scale=trunkScale, inverseMode=False)
    rootNode.trunk = trunk

    if visu:
        trunk.addVisualModel(color=[1., 1., 1., 0.8])
    trunk.fixExtremity()

    #print("config['goalPos'] in TrunkScene: ", config['goalPos'])


    pos_init = [0.0, 0.0, 0.0]
    pos_low = config["goal_low"]
    pos_high = config["goal_high"]
    add_box(rootNode, "goal", pos_low, pos_high, [1, 0, 1, 1]) #training box
    # if config["test"]:
    #     for g in config["goalList_test"]:
    #         if g is not config['goalPos']:
    #             add_box_point(rootNode, "goal_test_box", g, color=[0, 1, 0, 1])
        
    goal_mo, goal_visu = add_goal_node(rootNode, config['goalPos'])


    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos']))
    rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalVisu=goal_visu, goalPos=config['goalPos']))
    return rootNode
