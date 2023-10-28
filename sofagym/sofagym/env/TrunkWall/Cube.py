# -*- coding: utf-8 -*-
"""Create the Cube.

Units: cm, kg, s.
"""

__authors__ = ("emenager, cagabiti")
__contact__ = ("etienne.menager@inria.fr, Camilla.Agabiti@santannapisa.it")
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria, Biorobotics Institute"
__date__ = "Sept 28 2022"

import os
import numpy as np

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")


class Cube():
    def __init__(self, parentNode, totalMass, volume, inertialMatrix, position = [0, 0, 0], scale = [1, 1, 1]):
        self.scale = scale

        self.node = parentNode.addChild('Cube')
        self.node.addObject('MechanicalObject', name="mstate", template="Rigid3", showObject=True, showObjectScale=1,
                        translation=position, scale3d = scale, rotation=[0.0, 0.0, 0.0])
        self.node.addObject('UniformMass', name="vertexMass", vertexMass=[totalMass, volume, inertialMatrix[:]])

    def addVisualModel(self, color=[1., 1., 1., 1.]):
        cubeVisu = self.node.addChild('cubeVisu')
        cubeVisu.addObject("MeshOBJLoader", name="loader", filename = "mesh/smCube27.obj", scale3d= self.scale)
        cubeVisu.addObject('OglModel', name="Visual",src='@loader', color= color)
        cubeVisu.addObject('RigidMapping')

    def addCollisionModel(self, group = 1):
        cubeCollis = self.node.addChild('cubeCollis')
        cubeCollis.addObject('MeshOBJLoader', name="loader", filename="mesh/smCube27.obj", triangulate="true",  scale3d=self.scale)
        cubeCollis.addObject('MeshTopology', src="@loader")
        cubeCollis.addObject('MechanicalObject')
        cubeCollis.addObject('TriangleCollisionModel', group = group)
        cubeCollis.addObject('LineCollisionModel', group = group)
        cubeCollis.addObject('PointCollisionModel', group = group)
        cubeCollis.addObject('RigidMapping')


    def getPos(self):
        return self.node.mstate.position.value.tolist()

    def setPos(self, pos):
        self.node.mstate.position.value = np.array(pos)
