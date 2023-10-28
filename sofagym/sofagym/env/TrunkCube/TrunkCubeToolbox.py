# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
Developed as an extension of the original work https://github.com/SofaDefrost/SofaGym,
by adding Domain Randomization techinques.
"""

__authors__ = "Andrea Protopapa, Gabriele Tiboni"
__contact__ = "andrea.protopapa@polito.it, gabriele.tiboni@polito.it"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Politecnico di Torino, Italy"
__date__ = "Oct 28 2023"
import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib.animation.animate import Animation

SofaRuntime.importPlugin("SofaComponentAll")

def distance_penality_pos(d, alpha=0.006, beta= 500, gamma=1e-3):
        return -alpha*d**2 - beta*np.log(d**2+gamma)

def distance_penality_vel(d, alpha=0.6, beta=1e-3):
        return -alpha*d**2


class rewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.
        goal_pos: coordinates
            The position of the goal.
        effMO: <MechanicalObject>
            The mechanical object of the element to move.
        cost:
            Evolution of the distance between object and goal.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
        self.goal_pos = None
        if kwargs["goalPos"]:
            self.goal_pos = kwargs["goalPos"]
        self.effMO = None
        if kwargs["effMO"]:
            self.effMO = kwargs["effMO"]
        self.cost = None
        self.init_cost = None

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        #print("goal_pos from getReward: ", self.goal_pos)
        #print("GoalMO from getReward", self.rootNode.Goal.GoalMO.position[0])
        
        cube_position = np.array([self.effMO.position[0][0], self.effMO.position[0][2]]) # we consider only axis x and z
        goal_position = np.array([self.goal_pos[0], self.goal_pos[2]]) # we consider only axis x and z
        current_dist = np.linalg.norm(cube_position - goal_position) 
        #current_cost = current_dist**2
        current_cost = current_dist
        if not self.init_cost:
            self.cost = current_cost
            self.init_cost = current_cost
            return 0, self.cost

        # reward = self.cost - current_cost
        # self.cost = current_cost
        # return reward, current_cost

        #reward = max((self.cost - current_cost)/self.cost, 0)
        # self.cost = current_cost
        # return min(reward**(1/2), 1.0), current_cost

        # POSITION REWARD
        # reward_pos = (distance_penality(d=self.init_cost) - distance_penality(d=current_dist)) / distance_penality(d=self.init_cost) 
        reward_pos = (distance_penality_pos(d=self.init_cost) - distance_penality_pos(d=current_dist)) / distance_penality_pos(d=self.init_cost)
        reward_pos -= 0.05

        # VELOCITY REWARD
        #reward_vel = (distance_penality(d=self.cost) - distance_penality(d=current_dist)) / distance_penality(d=self.cost) 
        a = self.cost - current_dist
        reward_vel = distance_penality_vel(d=current_dist+a) - distance_penality_vel(d=current_dist)
        if a >= 0: # step near
            reward_vel /= distance_penality_vel(d=current_dist+a) 
        else: #step away
            reward_vel /= distance_penality_vel(d=current_dist) 

        self.cost = current_cost

        # alpha = 0.25
        # reward = alpha*reward_pos+(1-alpha)*reward_vel
        
        return reward_pos, current_cost

    def update(self):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.cost = np.linalg.norm(self.effMO.position[0][:3] - np.array(self.goal_pos))

    def _computeTips(self):
        """Compute the position of the tip.

        Parameters:
        ----------
            None.

        Return:
        ------
            The position of the tip.
        """
        cables = self.rootNode.trunk.cables[:4]
        size = len(cables)

        trunkTips = np.zeros(3)
        for cable in cables:
            trunkTips += cable.meca.position[-1]/size

        return trunkTips


class goalSetter(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        update: Initialize the value of cost.

    Arguments:
    ---------
        goalMO: <MechanicalObject>
            The mechanical object of the goal.
        goalVisu: <MeshOBJLoader>
            The viewable object of the goal.
        goalPos: coordinates
            The coordinates of the goal.

    """

    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.goalMO = None
        if kwargs["goalMO"]:
            self.goalMO = kwargs["goalMO"]
        self.goalVisu = None
        if kwargs["goalVisu"]:
            self.goalVisu = kwargs["goalVisu"]
        self.goalPos = None
        if kwargs["goalPos"]:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        """Set the position of the goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        print("In update()")
        with self.goalMO.position.writeable() as position:
            position = self.goalPos
            #position += self.goalPos
        with self.goalVisu.loader.translation.writeable() as translation:
            translation = self.goalPos
            #translation += self.goalPos
        print(f"\tgoal pos {self.goalPos}")
        print(f"\tposition goalMO {self.goalMO.position[0]}")
        print(f"\ttranslation {self.goalVisu.loader.translation.value}")

    def set_mo_pos(self, goal):
        """Modify the goal.

        Not used here.
        """
        pass


def _getGoalPos(rootNode):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return rootNode.Goal.GoalMO.position[0]


def getState(rootNode):
    """Compute the state of the environment/agent.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    cs = 3

    cube_pos = [round(float(k), cs) for k in rootNode.Simulation.Cube.mstate.position.value.reshape(-1)] # 7 DOFs

    cables = rootNode.trunk.cables[:4]
    nb_point = cables[0].meca.position.shape[0]

    points = [] # 21 x 3 DOFs = 63 DOFs
    for i in range(nb_point):
        point = np.zeros(3)
        for cable in cables:
            c = cable.meca.position[i]
            point += c
        point = [round(float(k), cs)/4 for k in point]
        points += point

    goalPos = _getGoalPos(rootNode).tolist() # 3 DOFs

    #print(f"toolbox goalPos from getState: {goalPos}")

    state = points + goalPos + cube_pos
    return state


def getReward(root):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """

    reward, cost = root.Reward.getReward()
    #print("cost ", cost)
    if cost <= 1.0:
        return True, reward, {"distance": float(cost)}

    return False, reward, {"distance": float(cost)} # done, reward, info dictionary


def startCmd(root, action, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    num_cable, displacement = action_to_command(action)

    # print(f"{num_cable} - {displacement}")
    # for i in range(len(root.trunk.cables)):
    #     print(f"Cable n. {i}")
    #     print(f"\tcableInitialLength: {root.trunk.cables[i].cable.cableInitialLength.getValueString()}")
    #     print(f"\tcableLength: {root.trunk.cables[i].cable.cableLength.getValueString()}")
    #     print(f"\tforce: {root.trunk.cables[i].cable.force.getValueString()}")
    #     print(f"\tdisplacement: {root.trunk.cables[i].cable.displacement.getValueString()}")
    #     print("\tDisplacement to impose: "+root.trunk.cables[i].cable.value.getValueString())

    startCmd_TrunkCube(root, root.trunk.cables[num_cable], displacement, duration)


def displace(cable, displacement):
    """Change the value of the cable in the finger.

    Parameters:
    ----------
        fingers:
            The finger.
        displacement: float
            The displacement.

    Returns:
    -------
        None.

    """
    cable.cable.value = [cable.cable.value[0] + displacement]


def startCmd_TrunkCube(rootNode, cable, displacement, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        cable: <MechanicalObject>
            The mechanical object of the cable to move.
        displacement: float
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(cable, displacement, factor):
        displace(cable, displacement)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"cable": cable,
                    "displacement": displacement},
            duration=duration, mode="once", realTimeClock=False))


def action_to_command(action):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).

    Returns:
    -------
        The command (number of the cabl and its displacement).
    """
    if action == 0:
        num_cable, displacement = 0, 1
    elif action == 1:
        num_cable, displacement = 1, 1
    elif action == 2:
        num_cable, displacement = 2, 1
    elif action == 3:
        num_cable, displacement = 3, 1
    elif action == 4:
        num_cable, displacement = 4, 1
    elif action == 5:
        num_cable, displacement = 5, 1
    elif action == 6:
        num_cable, displacement = 6, 1
    elif action == 7:
        num_cable, displacement = 7, 1
    elif   action == 8:
        num_cable, displacement = 0, -1
    elif action == 9:
        num_cable, displacement = 1, -1
    elif action == 10:
        num_cable, displacement = 2, -1
    elif action == 11:
        num_cable, displacement = 3, -1
    elif action == 12:
        num_cable, displacement = 4, -1
    elif action == 13:
        num_cable, displacement = 5, -1
    elif action == 14:
        num_cable, displacement = 6, -1
    elif action == 15:
        num_cable, displacement = 7, -1
    else:
        raise NotImplementedError("Action must be in range 0 - 15")

    return num_cable, displacement


def getPos(root):
    """Retun the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        _: list
            The position(s) of the object(s) of the scene.
    """
    trunk_pose = root.Simulation.Trunk.dofs.position.value.tolist()
    cube_pose = root.Simulation.Cube.mstate.position.value.tolist()

    return [trunk_pose, cube_pose]


def setPos(root, pos):
    """Set the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.
        pos: list
            The position(s) of the object(s) of the scene.

    Returns:
    -------
        None.

    Note:
    ----
        Don't forget to init the new value of the position.

    """
    [trunk_pose, cube_pose] = pos
    root.Simulation.Trunk.dofs.position.value = np.array(trunk_pose)
    root.Simulation.Cube.mstate.position.value = np.array(cube_pose)