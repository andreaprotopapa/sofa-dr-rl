# -*- coding: utf-8 -*-

def DefaultSolver(node, iterative=True):
    '''
    Adds EulerImplicit, CGLinearSolver

    Components added:
        EulerImplicitSolver
        CGLinearSolver
    '''
    node.addObject('EulerImplicitSolver', name='TimeIntegrationSchema')
    if iterative:
        return node.addObject('CGLinearSolver', name='LinearSolver')

    return node.addObject('SparseLDLSolver', name='LinearSolver', template='CompressedRowSparseMatrixd')

### This function is just an example on how to use the DefaultHeader function.
def createScene(rootNode):
	DefaultSolver(rootNode)
