# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 13:34:55 2014

@author: Noah
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd


"""
GOAL- to create a GA where initialization and objective functions are based on
the grid and everything else is based on translation between the grid and the 
data structure
"""

"""TODO
-Get rid of unneccesary references to dimension
-Implement checking in QuadTreeNode to ensure that the dimension is legitimate
-Create a simple framework for the implementation of different string ga's and objective functions


"""

def myPrint(myString,shouldIPrint=False):
    if shouldIPrint:
        print(myString)

#------------Data Structures------------------------------------
class QuadTreeNode(object):
    def __init__(self,grid,dimension,isRoot=False,ID=None,lowerLeftCoordinate = (0,0),depth=0,root=None,isLeaf=False,parent=None):        
        self.grid = grid
        self.isRoot = isRoot
        self.dimension = dimension #grid.shape[0]
        if self.dimension != 0 and not ((self.dimension & (self.dimension - 1)) == 0): # http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
            raise ValueError("The dimension should be a power of two")    
        self.ID = ID
        self.lowerLeftCoordinate = lowerLeftCoordinate
        self.depth = depth
        
        self.lowerLeftQuadrant = None
        self.upperLeftQuadrant = None
        self.upperRightQuadrant = None
        self.lowerRightQuadrant = None
        
        self.children = []
        self.parent = parent
        
        self.isLeaf = isLeaf
        self.nodeValueIfLeaf = None
        
        self.root = root
        
        if self.isRoot:
            self.ID = "Root"
            self.root = self
            self.leafNodes = []
            self.allChildren = []
    
    def checkIfHomogenous(self):
        isHomogenous = True
        valueToCheckAgainst = self.grid[self.lowerLeftCoordinate[1],self.lowerLeftCoordinate[0]]
        for x in range(self.lowerLeftCoordinate[0],self.lowerLeftCoordinate[0] + self.dimension):
            for y in range(self.lowerLeftCoordinate[1],self.lowerLeftCoordinate[1] + self.dimension):
                valueToCheck = self.grid[y,x]
                if valueToCheck != valueToCheckAgainst:
                    isHomogenous = False
                    return isHomogenous
        return isHomogenous
        
    def subdivide(self):
        isHomogenous = self.checkIfHomogenous()
        if isHomogenous:
            self.isLeaf = True
            self.nodeValueIfLeaf = self.grid[self.lowerLeftCoordinate[1],self.lowerLeftCoordinate[0]]
            self.root.leafNodes.append(self)
            self.root.allChildren.append(self)
        else:
            self.isLeaf = False
            halfDimension = self.dimension / 2
            
            # define the child quadrants
            llc = (self.lowerLeftCoordinate[0],self.lowerLeftCoordinate[1] + halfDimension)
            self.upperLeftQuadrant = QuadTreeNode(self.grid,halfDimension,lowerLeftCoordinate=llc,ID=self.ID + "->UpperLeftQuadrant",depth=self.depth+1,root=self.root,parent=self)
            
            llc = (self.lowerLeftCoordinate[0] + halfDimension,self.lowerLeftCoordinate[1]+halfDimension)
            self.upperRightQuadrant = QuadTreeNode(self.grid,halfDimension,lowerLeftCoordinate=llc,ID=self.ID + "->UpperRightQuadrant",depth=self.depth+1,root=self.root,parent=self)
            
            llc = (self.lowerLeftCoordinate[0] + halfDimension,self.lowerLeftCoordinate[1])
            self.lowerRightQuadrant = QuadTreeNode(self.grid,halfDimension,lowerLeftCoordinate=llc,ID=self.ID + "->LowerRightQuadrant",depth=self.depth+1,root=self.root,parent=self)
            
            self.lowerLeftQuadrant = QuadTreeNode(self.grid,halfDimension,lowerLeftCoordinate=self.lowerLeftCoordinate,ID=self.ID + "->LowerLeftQuadrant",depth=self.depth+1,root=self.root,parent=self)
            
            # repeat this process recursively
            self.upperLeftQuadrant.subdivide()
            self.upperRightQuadrant.subdivide()
            self.lowerRightQuadrant.subdivide()
            self.lowerLeftQuadrant.subdivide()
            
            # add the nodes to the child node
            self.children.append(self.upperLeftQuadrant)
            self.children.append(self.upperRightQuadrant)
            self.children.append(self.lowerRightQuadrant)
            self.children.append(self.lowerLeftQuadrant)
            self.root.allChildren.append(self.upperLeftQuadrant)
            self.root.allChildren.append(self.upperRightQuadrant)
            self.root.allChildren.append(self.lowerRightQuadrant)
            self.root.allChildren.append(self.lowerLeftQuadrant)
            
    def recreateGrid(self):
        grid = np.zeros((self.dimension,self.dimension))
        for leaf in self.leafNodes:
            for x in range(leaf.lowerLeftCoordinate[0],leaf.lowerLeftCoordinate[0]+leaf.dimension):
                for y in range(leaf.lowerLeftCoordinate[1],leaf.lowerLeftCoordinate[1]+leaf.dimension):
                    grid[y,x] = leaf.nodeValueIfLeaf
        return grid
            
    def createHash(self):
        quadTreeHash = ""
        for leaf in self.leafNodes:
            quadTreeHash += leaf.ID + "&"
        # remove the last '&'
        quadTreeHash = quadTreeHash.rstrip('&')
        self.quadTreeHash = quadTreeHash
        return quadTreeHash

#------------Data Structure Conversions-------------------------
def rasterToQuadTree(raster):
    return 0
    
def rasterToString(raster):
    return 0
    
def quadTreeToRaster(quadtree):
    return quadtree.recreateGrid()
    
def stringToRaster(string):
    return 0


#------------Raster-Based Operations----------------------------
def objectiveFunction(inputRaster):
    return 0
    


#------------Quadtree Genetic Operators-------------------------
def crossover(genome1,genome2,shouldIPrint=False):
    tree1 = genome1.quadtree
    tree2 = genome2.quadtree
    child1 = copy.deepcopy(tree1)
    child2 = copy.deepcopy(tree2)
    tree1IDs = [node.ID for node in tree1.allChildren]
    tree2IDs = [node.ID for node in tree2.allChildren]
    myPrint(tree1IDs,shouldIPrint)
    myPrint(tree2IDs,shouldIPrint)
    sharedTwoSet = list(set(tree1IDs).intersection(tree2IDs))
    myPrint(sharedTwoSet,shouldIPrint)
    # randomly select from this set
    crossoverPlace = np.random.randint(0,len(sharedTwoSet))
    crossoverNodeID = sharedTwoSet[crossoverPlace]
  
    # don't just swap the node, swap it and all children!
    child1ToSwap = copy.deepcopy([node for node in child1.allChildren if crossoverNodeID in node.ID])
    child1ToSwap_leafNodes = [node for node in child1ToSwap if node.isLeaf == True]
    child2ToSwap = copy.deepcopy([node for node in child2.allChildren if crossoverNodeID in node.ID])  
    child2ToSwap_leafNodes = [node for node in child2ToSwap if node.isLeaf == True]
    
    # delete from child1,child2 any nodes below the crossover point
    child1.allChildren = [node for node in child1.allChildren if crossoverNodeID not in node.ID]
    child1.leafNodes = [node for node in child1.leafNodes if crossoverNodeID not in node.ID]
    child2.allChildren = [node for node in child2.allChildren if crossoverNodeID not in node.ID]    
    child2.leafNodes = [node for node in child2.leafNodes if crossoverNodeID not in node.ID]
    
    # append the swap point to the child list
    child1.allChildren.extend(child2ToSwap)
    child2.allChildren.extend(child1ToSwap)
    child1.leafNodes.extend(child2ToSwap_leafNodes)
    child2.leafNodes.extend(child1ToSwap_leafNodes)
    
    child1Genome = Genome()
    child1Genome.quadtree = child1
    child2Genome = Genome()
    child2Genome.quadtree = child2
    
    return child1Genome,child2Genome
    
    
def mutation_alteration(genome,pMa,possibleSet):
    tree = genome.quadtree
    mutatedTree = copy.deepcopy(tree)
    for node in mutatedTree.leafNodes:
        doWeMutate = np.random.random() < pMa
        if doWeMutate:
            mutationIndex = np.random.randint(0,len(possibleSet))
            node.nodeValueIfLeaf = possibleSet[mutationIndex]
    mutatedGenome = Genome()
    mutatedGenome.quadtree = mutatedTree
    return mutatedGenome 
    
    
def mutation_splitting(genome,pMs,possibleSet):
    """
    Only consider splitting leaf nodes which are splittable (e.g. dimension > 1)    
    """
    tree = genome.quadtree
    mutatedTree = copy.deepcopy(tree)
    for node in [splittable for splittable in mutatedTree.leafNodes if splittable.dimension > 1]:
        doWeMutate = np.random.random() < pMs
        if doWeMutate:
            node.isLeaf = False
            node.root.leafNodes = [leaf for leaf in node.root.leafNodes if leaf.ID != node.ID]
            
            halfDimension = node.dimension / 2
            
            # set each child quadrant
            llc = (node.lowerLeftCoordinate[0],node.lowerLeftCoordinate[1] + halfDimension)
            node.upperLeftQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->UpperLeftQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
            quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
            node.upperLeftQuadrant.nodeValueIfLeaf = quadrantValue            
            
            llc = (node.lowerLeftCoordinate[0] + halfDimension,node.lowerLeftCoordinate[1]+halfDimension)
            node.upperRightQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->UpperRightQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
            quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
            node.upperRightQuadrant.nodeValueIfLeaf = quadrantValue              
            
            llc = (node.lowerLeftCoordinate[0] + halfDimension,node.lowerLeftCoordinate[1])
            node.lowerRightQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->LowerRightQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
            quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
            node.lowerRightQuadrant.nodeValueIfLeaf = quadrantValue              
            
            node.lowerLeftQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=node.lowerLeftCoordinate,ID=node.ID + "->LowerLeftQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
            quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
            node.lowerLeftQuadrant.nodeValueIfLeaf = quadrantValue         
                        
            # add the nodes to the child node
            node.children.append(node.upperLeftQuadrant)
            node.children.append(node.upperRightQuadrant)
            node.children.append(node.lowerRightQuadrant)
            node.children.append(node.lowerLeftQuadrant)
            node.root.allChildren.append(node.upperLeftQuadrant)
            node.root.allChildren.append(node.upperRightQuadrant)
            node.root.allChildren.append(node.lowerRightQuadrant)
            node.root.allChildren.append(node.lowerLeftQuadrant)            
            node.root.leafNodes.append(node.upperLeftQuadrant)
            node.root.leafNodes.append(node.upperRightQuadrant)
            node.root.leafNodes.append(node.lowerRightQuadrant)
            node.root.leafNodes.append(node.lowerLeftQuadrant)
    
    mutatedGenome = Genome()
    mutatedGenome.quadtree = mutatedTree
    return mutatedGenome


def mutation_splitting_twoResolution(genome,pMs,possibleSet):
    """
    Only consider splitting leaf nodes which are splittable (e.g. dimension > 1)    
    """
    tree = genome.quadtree
    mutatedTree = copy.deepcopy(tree)
    for node in [splittable for splittable in mutatedTree.leafNodes if splittable.dimension > 1]:
        doWeMutate = np.random.random() < pMs
        if doWeMutate:
            increaseOrDecreaseResolution = np.random.randint(0,2)
            if increaseOrDecreaseResolution == 0:
                # Increase resolution
                node.isLeaf = False
                node.root.leafNodes = [leaf for leaf in node.root.leafNodes if leaf.ID != node.ID]
                
                halfDimension = node.dimension / 2
                
                # set each child quadrant
                llc = (node.lowerLeftCoordinate[0],node.lowerLeftCoordinate[1] + halfDimension)
                node.upperLeftQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->UpperLeftQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
                quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
                node.upperLeftQuadrant.nodeValueIfLeaf = quadrantValue            
                
                llc = (node.lowerLeftCoordinate[0] + halfDimension,node.lowerLeftCoordinate[1]+halfDimension)
                node.upperRightQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->UpperRightQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
                quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
                node.upperRightQuadrant.nodeValueIfLeaf = quadrantValue              
                
                llc = (node.lowerLeftCoordinate[0] + halfDimension,node.lowerLeftCoordinate[1])
                node.lowerRightQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=llc,ID=node.ID + "->LowerRightQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
                quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
                node.lowerRightQuadrant.nodeValueIfLeaf = quadrantValue              
                
                node.lowerLeftQuadrant = QuadTreeNode(node.grid,halfDimension,lowerLeftCoordinate=node.lowerLeftCoordinate,ID=node.ID + "->LowerLeftQuadrant",depth=node.depth+1,root=node.root,isLeaf=True)
                quadrantValue = possibleSet[np.random.randint(0,len(possibleSet))]
                node.lowerLeftQuadrant.nodeValueIfLeaf = quadrantValue         
                            
                # add the nodes to the child node
                node.children.append(node.upperLeftQuadrant)
                node.children.append(node.upperRightQuadrant)
                node.children.append(node.lowerRightQuadrant)
                node.children.append(node.lowerLeftQuadrant)
                node.root.allChildren.append(node.upperLeftQuadrant)
                node.root.allChildren.append(node.upperRightQuadrant)
                node.root.allChildren.append(node.lowerRightQuadrant)
                node.root.allChildren.append(node.lowerLeftQuadrant)            
                node.root.leafNodes.append(node.upperLeftQuadrant)
                node.root.leafNodes.append(node.upperRightQuadrant)
                node.root.leafNodes.append(node.lowerRightQuadrant)
                node.root.leafNodes.append(node.lowerLeftQuadrant)
            
        else:
            # decrease resolution
            if node.parent != None:
                newLeaf = node.parent
                for child in newLeaf.children:
                    child.isLeaf = False
                    # remove the ID from the child nodes
                    node.root.leafNodes = [leaf for leaf in node.root.leafNodes if leaf.ID != child.ID]
                newLeaf.children = []
                newValue = np.random.randint(0,len(possibleSet))
                newLeaf.nodeValueIfLeaf = possibleSet[newValue]
                newLeaf.isLeaf = True
                
    mutatedGenome = Genome()
    mutatedGenome.quadtree = mutatedTree
    return mutatedGenome

#------------String Genetic Operators---------------------------











  

    
#----------GA Construction---------------------------------

class Genome(object):
    def __init__(self):
        self.quadtree = None
        self.score = None
        self.rouletteFloor = None
        self.rouletteCeiling = None

def initializePopulation(populationSize,dimension,possibleSet):
    population = []
    for i in range(0,populationSize):
        genome = Genome()
        initialGrid = np.zeros((dimension,dimension))
        for x in range(0,dimension):
            for y in range(0,dimension):
                initialValue = possibleSet[np.random.randint(0,len(possibleSet))]
                initialGrid[y,x] = initialValue
        root = QuadTreeNode(initialGrid,dimension,isRoot=True)
        root.subdivide()
        genome.quadtree = root
        population.append(genome)
    return population
    
def initializePopulation_2(populationSize,dimension,possibleSet):
    population = []
    for i in range(0,populationSize):
        # the number of leafs is a power of 4 up to 2**x=dimension
        maxDepth = int(np.math.log(dimension,2))
        genome = Genome()
        depth = np.random.randint(0,maxDepth+1)
        leafSize = dimension / 2 - depth
        nLeafs = dimension**2 / leafSize        
        
        
        genome = Genome()
        initialGrid = np.zeros((dimension,dimension))
        for x in range(0,dimension):
            for y in range(0,dimension):
                initialValue = possibleSet[np.random.randint(0,len(possibleSet))]
                initialGrid[y,x] = initialValue
        root = QuadTreeNode(initialGrid,dimension,isRoot=True)
        root.subdivide()
        genome.quadtree = root
        population.append(genome)
    return population
    
def initializePopulation_3(populationSize,dimension,possibleSet):
    population = []
    for i in range(0,populationSize):
        # create four leaf nodes
        # for each leaf node, consider breaking it up
        
        genome = Genome()
        initialGrid = np.zeros((dimension,dimension))
        for x in range(0,dimension):
            for y in range(0,dimension):
                initialValue = possibleSet[np.random.randint(0,len(possibleSet))]
                initialGrid[y,x] = initialValue
        root = QuadTreeNode(initialGrid,dimension,isRoot=True)
        root.subdivide()
        genome.quadtree = root
        population.append(genome)
    return population
    
def initializePopulation_4(populationSize,dimension,possibleSet):
    population = []
    for i in range(0,populationSize):
        genome = Genome()
        # create just four leaf nodes
        halfDimension = dimension / 2
        upperLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperLeftQuadrant = np.ones((halfDimension,halfDimension)) * upperLeftQuadrantValue
        
        upperRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperRightQuadrant = np.ones((halfDimension,halfDimension)) * upperRightQuadrantValue
        
        lowerLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerLeftQuadrant = np.ones((halfDimension,halfDimension)) * lowerLeftQuadrantValue
        
        lowerRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerRightQuadrant = np.ones((halfDimension,halfDimension)) * lowerRightQuadrantValue
    
        gridLeft = np.concatenate((upperLeftQuadrant,lowerLeftQuadrant),axis=0)
        gridRight = np.concatenate((upperRightQuadrant,lowerRightQuadrant),axis=0)
        grid = np.concatenate((gridLeft,gridRight),axis=1)
        
        
        root = QuadTreeNode(grid,dimension,isRoot=True)
        root.subdivide()
        genome.quadtree = root
        population.append(genome)
    return population
    
def initializePopulation_5(populationSize,dimension,possibleSet):
    population = []
    for i in range(0,populationSize):
        genome = Genome()
        # create just leaf nodes
        
        
        quarterDimension = dimension / 4
        
        upperLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * upperLeftQuadrantValue
        
        upperRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperRightQuadrant = np.ones((quarterDimension,quarterDimension)) * upperRightQuadrantValue
        
        lowerLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerLeftQuadrantValue
        
        lowerRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerRightQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerRightQuadrantValue
    
        gridLeft = np.concatenate((upperLeftQuadrant,lowerLeftQuadrant),axis=0)
        gridRight = np.concatenate((upperRightQuadrant,lowerRightQuadrant),axis=0)
        grid1 = np.concatenate((gridLeft,gridRight),axis=1)
        
        
        upperLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * upperLeftQuadrantValue
        
        upperRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperRightQuadrant = np.ones((quarterDimension,quarterDimension)) * upperRightQuadrantValue
        
        lowerLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerLeftQuadrantValue
        
        lowerRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerRightQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerRightQuadrantValue
    
        gridLeft = np.concatenate((upperLeftQuadrant,lowerLeftQuadrant),axis=0)
        gridRight = np.concatenate((upperRightQuadrant,lowerRightQuadrant),axis=0)
        grid2 = np.concatenate((gridLeft,gridRight),axis=1)
        
        
        upperLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * upperLeftQuadrantValue
        
        upperRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperRightQuadrant = np.ones((quarterDimension,quarterDimension)) * upperRightQuadrantValue
        
        lowerLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerLeftQuadrantValue
        
        lowerRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerRightQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerRightQuadrantValue
    
        gridLeft = np.concatenate((upperLeftQuadrant,lowerLeftQuadrant),axis=0)
        gridRight = np.concatenate((upperRightQuadrant,lowerRightQuadrant),axis=0)
        grid3 = np.concatenate((gridLeft,gridRight),axis=1)
        
        
        upperLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * upperLeftQuadrantValue
        
        upperRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        upperRightQuadrant = np.ones((quarterDimension,quarterDimension)) * upperRightQuadrantValue
        
        lowerLeftQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerLeftQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerLeftQuadrantValue
        
        lowerRightQuadrantValue = possibleSet[np.random.randint(len(possibleSet))]
        lowerRightQuadrant = np.ones((quarterDimension,quarterDimension)) * lowerRightQuadrantValue
    
        gridLeft = np.concatenate((upperLeftQuadrant,lowerLeftQuadrant),axis=0)
        gridRight = np.concatenate((upperRightQuadrant,lowerRightQuadrant),axis=0)
        grid4 = np.concatenate((gridLeft,gridRight),axis=1)
        
        
        gridLeft = np.concatenate((grid1,grid2),axis=0)
        gridRight = np.concatenate((grid3,grid4),axis=0)
        grid = np.concatenate((gridLeft,gridRight),axis=1)        
        root = QuadTreeNode(grid,dimension,isRoot=True)
        root.subdivide()
        genome.quadtree = root
        population.append(genome)
    return population
    
def objectiveFunction(genome,solutionSpace):
    quadTreeScore = 0
    solnGrid = genome.quadtree.recreateGrid()
    for x in range(0,genome.quadtree.dimension):
        for y in range(0,genome.quadtree.dimension):
            cellScore = solutionSpace[solnGrid[y,x],y,x] # verify this
            quadTreeScore += cellScore
    genome.score = quadTreeScore
    return quadTreeScore

def evaluateAll(population,solutionSpace,shouldIPrint=False):
    totalScore = 0
    for genome in population:
        score = objectiveFunction(genome,solutionSpace)
        totalScore += score
    rouletteFloor = 0.0
    for genome in population:
        genome.rouletteFloor = rouletteFloor
        delta = float(genome.score) / float(totalScore)
        rouletteFloor += delta
        genome.rouletteCeiling = rouletteFloor
    averageScore = float(totalScore) / float(len(population))
#    myPrint("AverageScore: " + str(averageScore),shouldIPrint=True)
    return averageScore

def selection_roulette(population):
    roulette = np.random.random()
    for genome in population:
        if genome.rouletteFloor <= roulette <= genome.rouletteCeiling:
            sel = copy.deepcopy(genome)
            return sel
            
class Analysis(object):
    def __init__(self):
        self.populations = []
        self.averageScores = []
        self.solutionSpace = None
        self.parameters = {}
        self.topScores = []


        
#----------Perform GA--------------------------------------
def GA_roulette(solutionSpace,dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint=False,elitism=False):
    reporter = Analysis()
    reporter.parameters['solutionSpace'] = solutionSpace
    reporter.parameters['dimension'] = dimension
    reporter.parameters['possibleSet'] = possibleSet
    reporter.parameters['populationSize'] = populationSize
    reporter.parameters['nGenerations'] = nGenerations
    reporter.parameters['pC'] = pC
    reporter.parameters['pMa'] = pMa
    reporter.parameters['pMs'] = pMs
    reporter.solutionSpace = solutionSpace
    population = initializePopulation_5(populationSize,dimension,possibleSet)
    for generation in range(0,nGenerations):
        myPrint("Generation: " + str(generation),shouldIPrint)
        reporter.populations.append(population)
        avgScore = evaluateAll(population,solutionSpace,shouldIPrint)
        reporter.averageScores.append(avgScore)
        nextPopulation = []
        if elitism:
            elitePopulation = copy.deepcopy(population)
            elitePopulation.sort(key=lambda x:x.score, reverse=True)
#            print elitePopulation[0].score
            nextPopulation.append(copy.deepcopy(elitePopulation[0]))
            nextPopulation.append(copy.deepcopy(elitePopulation[0]))
        while len(nextPopulation) < len(population):
            shouldWeCrossover = np.random.random() < pC
            if shouldWeCrossover:
                sel1 = selection_roulette(population)
                sel2 = selection_roulette(population)
                child1,child2 = crossover(sel1,sel2,shouldIPrint)
                nextPopulation.append(child1)
                nextPopulation.append(child2)
            else:
                sel1 = selection_roulette(population)
                nextPopulation.append(sel1)
                sel2 = selection_roulette(population)
                nextPopulation.append(sel2)
        
        # mutation
        for genome in nextPopulation:
            shouldWeAlter = np.random.random() < pMa
            if shouldWeAlter:
                genome = mutation_alteration(genome,0.01,possibleSet)
            shouldWeSplit = np.random.random() < pMs
            if shouldWeSplit:
                genome = mutation_splitting_twoResolution(genome,0.01,possibleSet)
                
        popScores = population
        popScores.sort(key=lambda x:x.score, reverse=True)
        topScore = popScores[0].score
        reporter.topScores.append(topScore)
        
        population = nextPopulation
    return reporter
    
def punishingObjectiveFunction(genome,solutionSpace):
    quadTreeScore = 0
    solnGrid = genome.quadtree.recreateGrid()
    for x in range(0,genome.quadtree.dimension):
        for y in range(0,genome.quadtree.dimension):
            cellScore = solutionSpace[solnGrid[y,x],y,x] # verify this
            quadTreeScore += cellScore
    genome.score = quadTreeScore
    
    # now apply punishment for clustered diversity
    
    return quadTreeScore
    
def GA(elitism=False):
#    solutionSpace = np.array([[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]],[[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]])
#    solutionSpace = np.array([[[10,10,1,1],[10,10,1,1],[1,1,10,10],[1,1,10,10]],[[1,1,10,10],[1,1,10,10],[10,10,1,1],[10,10,1,1]]])  
#    print("Solution Space\n" + str(solutionSpace))
    solutionSpace = np.array([[[10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10],
                               [10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10]],

                      [[1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1],
                       [1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1]]])  
                       
    print(solutionSpace)
    dimension = 8
    possibleSet = [0,1]
    populationSize = 100
    nGenerations = 5
    pC = 0.7
    pMa = 0.5
    pMs = 0.7
    shouldIPrint = False
    reporter = GA_roulette(solutionSpace,dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint,elitism)
    # pick one of the top scoring individuals
    toPrint = reporter.populations[-1][0].quadtree
    print(toPrint.recreateGrid())
    averageScore = max(reporter.averageScores)
    averageScores = reporter.averageScores
    plt.title("Non-Epistatic Objective Function:\nQuadtree Encoding\npopSize:"+str(populationSize)+",nGen:"+str(nGenerations)+"\npC:"+str(pC)+",pMa:"+str(pMa)+",pMs:"+str(pMs)+",elitism:"+str(elitism)+"\nAverageScores:" + str(averageScores))
    plt.imshow(toPrint.recreateGrid())
    plt.show()
    return reporter



#-----------Testing Operators------------------------------


def testCrossover():
#    grid1 = np.ones((8,8))
#    grid2 = np.zeros((8,8))

#    grid1 = np.random.randint(0,2,(8,8))
#    grid2 = np.random.randint(2,4,(8,8))

#    grid1 = np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])
#    grid2 = np.array([[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]])

#    grid1 = np.array([[1,1,2,2,3,3,4,4],
#                      [1,1,2,2,3,3,4,4],
#                      [5,5,6,6,7,7,8,8],
#                      [5,5,6,6,7,7,8,8],
#                      [1,1,2,2,3,3,4,4],
#                      [1,1,2,2,3,3,4,4],
#                      [5,5,6,6,7,7,8,8],
#                      [5,5,6,6,7,7,8,8]])
#    grid2 = copy.deepcopy(grid1)
#    grid2 = np.add(grid2,10)

#    grid1 = np.random.randint(0,2,(64,64))
#    grid2 = np.random.randint(2,4,(64,64))    

    grid1 = np.array([[1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8],
                      [1,2,3,4,5,6,7,8]])
    grid2 = copy.deepcopy(grid1)
    grid2 = np.add(grid2,10)
    
    
    tree1 = QuadTreeNode(grid1,8,isRoot=True)
    tree2 = QuadTreeNode(grid2,8,isRoot=True)
    tree1.subdivide()
    tree2.subdivide()
    print tree1.recreateGrid()
    print tree2.recreateGrid()
    print tree1.allChildren
    print tree2.allChildren
    child1,child2 = crossover(tree1,tree2)
    print child1.recreateGrid()
    print child2.recreateGrid()
    plt.figure()
    plt.imshow(tree1.recreateGrid())
    plt.figure()
    plt.imshow(tree2.recreateGrid())
    plt.figure()
    plt.imshow(child1.recreateGrid())
    plt.figure()
    plt.imshow(child2.recreateGrid())



def test_mutation_alteration():
#    grid = np.zeros((8,8))

    grid = np.array([[ 2,  5, 10, 17, 17, 10,  5,  2],
                     [ 5,  8, 13, 20, 20, 13,  8,  5],
                     [10, 13, 18, 25, 25, 18, 13, 10],
                     [17, 20, 25, 32, 32, 25, 20, 17],
                     [17, 20, 25, 32, 32, 25, 20, 17],
                     [10, 13, 18, 25, 25, 18, 13, 10],
                     [ 5,  8, 13, 20, 20, 13,  8,  5],
                     [ 2,  5, 10, 17, 17, 10,  5,  2]])


    possibleSet = [-1,-2,-3,-4]
    tree = QuadTreeNode(grid,8,isRoot=True)
    tree.subdivide()
    mutatedTree = mutation_alteration(tree,0.05,possibleSet)
    print mutatedTree.recreateGrid()
    plt.imshow(mutatedTree.recreateGrid())

def test_mutation_splitting():
#    grid = np.zeros((8,8))

#    grid = np.array([[ 2,  5, 10, 17, 17, 10,  5,  2],
#                     [ 5,  8, 13, 20, 20, 13,  8,  5],
#                     [10, 13, 18, 25, 25, 18, 13, 10],
#                     [17, 20, 25, 32, 32, 25, 20, 17],
#                     [17, 20, 25, 32, 32, 25, 20, 17],
#                     [10, 13, 18, 25, 25, 18, 13, 10],
#                     [ 5,  8, 13, 20, 20, 13,  8,  5],
#                     [ 2,  5, 10, 17, 17, 10,  5,  2]])

#    grid = np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])

    grid = np.array([[1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [5,5,6,6,7,7,8,8],
                      [5,5,6,6,7,7,8,8],
                      [1,1,2,2,3,3,4,4],
                      [1,1,2,2,3,3,4,4],
                      [5,5,6,6,7,7,8,8],
                      [5,5,6,6,7,7,8,8]])
                     
    possibleSet = [-1,-2,-3,-4]
    tree = QuadTreeNode(grid,4,isRoot=True)
    tree.subdivide()
    mutatedTree = mutation_splitting(tree,0.3,possibleSet)
    print mutatedTree.recreateGrid()
    plt.imshow(mutatedTree.recreateGrid())





# epistasis tests
def createEpistaticObjectiveFunction():
    objectiveFunction = {} # {leafID,Score}
    
    # code to combine quadrants into an array
#    a = np.random.randint(1,3,(4,4))
#    b = np.concatenate((a,a),axis=0)
#    b = np.concatenate((b,b),axis=1)
    
    # one leaf
    grid = np.ones((8,8)) 
#    print grid
    root = QuadTreeNode(grid,8,isRoot=True)
    root.subdivide()
    for leaf in root.leafNodes:
        ID = leaf.ID
        objectiveFunction[ID] = 1000
    
    # four leaf
    grid1 = np.ones((4,4))
    grid2 = np.ones((4,4)) * 2
    grid3 = np.ones((4,4)) * 3
    grid4 = np.ones((4,4)) * 4
    gridLeft = np.concatenate((grid1,grid2),axis=0)
    gridRight = np.concatenate((grid3,grid4),axis=0)
    grid = np.concatenate((gridLeft,gridRight),axis=1)
#    print grid
    root = QuadTreeNode(grid,8,isRoot=True)
    root.subdivide()
    for leaf in root.leafNodes:
        ID = leaf.ID
        objectiveFunction[ID] = 100    
    
#    # 16 leaf
#    quadrants = np.zeros((2,2))
#    for i in range(0,16):
#        multiplier = i + 1
#        quadrant = np.ones((2,2)) * multiplier
#        quadrants = np.concatenate((quadrants,quadrant),axis=0)
#    quadrants = quadrants[2:]
#    newQuadrants = np.concatenate((quadrants[:16],quadrants[16:]),axis=1)
#    grid = np.concatenate((newQuadrants[:8],newQuadrants[8:]),axis=1)
##    print newNewQuadrants
#    root = QuadTreeNode(grid,8,isRoot=True)
#    root.subdivide()
#    for leaf in root.leafNodes:
#        ID = leaf.ID
#        objectiveFunction[ID] = 100
        
# Kludge
#    objectiveFunction['Root->UpperRightQuadrant->LowerLeftQuadrant'] = 100
#    objectiveFunction['Root->UpperRightQuadrant->LowerRightQuadrant'] =  100
#    objectiveFunction['Root->UpperRightQuadrant->UpperLeftQuadrant'] = 100
#    objectiveFunction['Root->UpperRightQuadrant->UpperRightQuadrant'] = 100
    
    #64 leaf
    grid = np.zeros((8,8))
    value = 1
    for x in range(0,8):
        for y in range(0,8):
            grid[y,x] = value
            value += 1
#    print grid
            
    return objectiveFunction
    
def createEpistaticObjectiveFunction2():
    objectiveFunction = {} # {leafID,Score}
    solutionSpace0 = np.array([[10,10,1,1],[10,10,1,1],[1,1,10,10],[1,1,10,10]])  
    solutionSpace1 = np.array([[1,1,10,10],[1,1,10,10],[10,10,1,1],[10,10,1,1]]) 
    root1 = QuadTreeNode(solutionSpace1,4,isRoot=True)
    root1.subdivide()
    for leaf in root1.leafNodes:
        ID = leaf.ID
        if leaf.nodeValueIfLeaf == 10:
            objectiveFunction[ID] = 100
        else:
            objectiveFunction[ID] = 0
    return objectiveFunction
    


    
def epistaticObjectiveFunction(objectiveFunction,solnGrid):
#    print solnGrid
    dim = solnGrid.shape[0]
    root = QuadTreeNode(solnGrid,dim,isRoot=True)
    root.subdivide()
    score = 0
    for leaf in root.leafNodes:
        ID = leaf.ID
        if objectiveFunction.has_key(ID):
#            print ID
#            print "Found"
            score += objectiveFunction[ID]
        else:
            score += 1
    return score
    
def evaluatePopulation_epistatic(population):
    objectiveFunction = createEpistaticObjectiveFunction()
    totalScore = 0.0
    topScore = 0.0
    topIndividual = None
    for individual in population:
        solnGrid  = individual.quadtree.recreateGrid()
        score = epistaticObjectiveFunction(objectiveFunction,solnGrid)
        individual.score = score
        if score > topScore:
            topScore = score
            topIndividual = individual
        totalScore += score
#        print(str(individual.bits) + ": " + str(score))
#        print("Score: " + str(score))
    rouletteFloor = 0.0
    for individual in population:
        individual.rouletteFloor = rouletteFloor
        rouletteFloor += individual.score / totalScore
        individual.rouletteCeiling = rouletteFloor
        
    averageScore = totalScore / len(population)
    print("Average Score: " + str(averageScore))
#    plt.figure()
#    plt.imshow(topIndividual.quadtree.recreateGrid())
#    plt.show()
    return averageScore,topScore,topIndividual
    
def GA_roulette_epistatic(dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint=False,elitism=False):
    reporter = Analysis()
    reporter.parameters['dimension'] = dimension
    reporter.parameters['possibleSet'] = possibleSet
    reporter.parameters['populationSize'] = populationSize
    reporter.parameters['nGenerations'] = nGenerations
    reporter.parameters['pC'] = pC
    reporter.parameters['pMa'] = pMa
    reporter.parameters['pMs'] = pMs
    population = initializePopulation_5(populationSize,dimension,possibleSet)
    for generation in range(0,nGenerations):
#        myPrint("Generation: " + str(generation),True)
        reporter.populations.append(population)
        avgScore,topScore,topIndividual = evaluatePopulation_epistatic(population)
        reporter.averageScores.append(avgScore)
        reporter.topScores.append(topScore)
        
        # plot the best individual
#        plt.figure()
#        plt.title("Quadtree Genetic Algorithm\nGeneration: " + str(generation) + "\npopSize:"+str(populationSize)+",nGen:"+str(nGenerations)+"\npC:"+str(pC)+",pMa:"+str(pMa)+",pMs:"+str(pMs)+",elitism:"+str(elitism))
#        plt.imshow(topIndividual.quadtree.recreateGrid())
#        plt.show()
        
        nextPopulation = []
        if elitism:
            elitePopulation = copy.deepcopy(population)
            elitePopulation.sort(key=lambda x:x.score, reverse=True)
#            print elitePopulation[0].score
            nextPopulation.append(copy.deepcopy(elitePopulation[0]))
            nextPopulation.append(copy.deepcopy(elitePopulation[0]))
        while len(nextPopulation) < len(population):
            shouldWeCrossover = np.random.random() < pC
            if shouldWeCrossover:
                sel1 = selection_roulette(population)
                sel2 = selection_roulette(population)
                child1,child2 = crossover(sel1,sel2,shouldIPrint)
                nextPopulation.append(child1)
                nextPopulation.append(child2)
            else:
                sel1 = selection_roulette(population)
                nextPopulation.append(sel1)
                sel2 = selection_roulette(population)
                nextPopulation.append(sel2)
        
        # mutation
        for genome in nextPopulation:
            shouldWeAlter = np.random.random() < pMa
            if shouldWeAlter:
                genome = mutation_alteration(genome,0.01,possibleSet)
            shouldWeSplit = np.random.random() < pMs
            if shouldWeSplit:
                genome = mutation_splitting_twoResolution(genome,0.01,possibleSet)
                
#        popScores = population
#        popScores.sort(key=lambda x:x.score, reverse=True)
#        topScore = popScores[0].score
#        reporter.topScores.append(topScore)
#        
        population = nextPopulation
    return reporter

def GA_epistatic(elitism=False):                      
    dimension = 128
    possibleSet = [0,1]
    populationSize = 100
    nGenerations = 5
    pC = 0.7
    pMa = 0.5
    pMs = 0.7
    shouldIPrint = False
    reporter = GA_roulette_epistatic(dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint,elitism)
    # pick one of the top scoring individuals
    toPrint = reporter.populations[-1][0].quadtree
    print(toPrint.recreateGrid())
    averageScore = max(reporter.averageScores)
    averageScores = reporter.averageScores
    plt.title("Epistatic Objective Function:\nQuadtree Encoding\npopSize:"+str(populationSize)+",nGen:"+str(nGenerations)+"\npC:"+str(pC)+",pMa:"+str(pMa)+",pMs:"+str(pMs)+",elitism:"+str(elitism)+"\nAverageScores:" + str(averageScores))
    plt.imshow(toPrint.recreateGrid())
    plt.show()
    return reporter
    
def GA_MC():
    dimension = 8
    possibleSet = [0,1]
#    populationSizes = [10,20,50,100]
#    nGens = [1,2,5,10,50]
#    pCs = [.5,.6,.7,.8,.9,1]
#    pMas = [0.1,0.3,0.5,0.7,0.9]
#    pMss = [0.1,0.3,0.5,0.7,0.9]
    populationSizes = [10,20,50]
    nGens = [5,10]
    pCs = [0.5,0.7,0.9]
    pMas = [0.1,0.3,0.5,0.7]
    pMss = [0.1,0.3,0.5,0.7,0.9]
    elitisms = [True,False]
    mc = 2
    
    results = pd.DataFrame (np.zeros(0,dtype=[('populationSize','i4'),('nGen','i4'),('pC','i4'),('pMa','i4'),('pMs','i4'),('elitism','i4'),('bestScore','i4')]))
    # courtesy of http://technicaltidbit.blogspot.com/2013/06/create-empty-dataframe-in-pandas.html
    
    test = 1
    nTestsRequired = len(populationSizes)*len(nGens)*len(pCs)*len(pMas)*len(pMss)*len(elitisms)*mc
    for populationSize in populationSizes:
        for nGen in nGens:
            for pC in pCs:
                for pMa in pMas:
                    for pMs in pMss:
                        for elitism in elitisms:
                            for i in range(0,mc):
                                shouldIPrint = False
                                reporter = GA_roulette_epistatic(dimension,possibleSet,populationSize,nGen,pC,pMa,pMs,shouldIPrint,elitism)
                                bestScore = np.max(reporter.topScores)
#                                print("Test: " + str(test) + " of " + str(nTestsRequired) + ",BestScore: " + str(bestScore) + ", Params: popSize:" + str(populationSize)+",nGen:"+str(nGen)+",pC:"+str(pC)+",pMa:"+str(pMa)+",pMs:"+str(pMs)+",elitism:"+str(elitism))
                                results = results.append({'populationSize':populationSize,'nGen':nGen,'pC':pC,'pMa':pMa,'pMs':pMs,'elitism':elitism,'bestScore':bestScore},ignore_index=True)
                                if test % 20 == 0:
                                    print("Test: " + str(test) + " of " + str(nTestsRequired))
                                test += 1
    results.to_csv("QTGA_MC.csv")
    import statsmodels.formula.api as sm
    result = sm.ols(formula="bestScore~populationSize+nGen+pC+pMa+pMs+elitism",data=results).fit()
    print result.summary()
    return results

def evaluateRandom_epistatic(nRuns,rasterDim):
    # tests for a random raster with two possible values, 1 and 0
    objectiveFunction = createEpistaticObjectiveFunction()
    results = {}
    resultsArray = []
    for i in range(0,nRuns):
        newHash = False
        while not newHash:
            randomSoln = np.random.randint(0,2,(rasterDim,rasterDim))
            solnHash = np.array_str(randomSoln)          
            if solnHash not in results:
                newHash = True
        score = epistaticObjectiveFunction(objectiveFunction,randomSoln)
        results[solnHash] = score
        resultsArray.append(score)
        
    return resultsArray
    

def objectiveFunction_Random(solnGrid,solutionSpace):
    solnScore = 0
    for x in range(0,solnGrid.shape[0]):
        for y in range(0,solnGrid.shape[0]):
            cellScore = solutionSpace[solnGrid[y,x],y,x] # verify this
            solnScore += cellScore
    return solnScore


def evaluateRandom(nRuns,rasterDim):
    solutionSpace = np.array([[[10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10],
                               [10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10]],

                      [[1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1],
                       [1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1]]])  
    
    results = {}
    resultsArray = []
    for i in range(0,nRuns):
        newHash = False
        while not newHash:
            randomSoln = np.random.randint(0,2,(rasterDim,rasterDim))
            solnHash = np.array_str(randomSoln)          
            if solnHash not in results:
                newHash = True
        score = objectiveFunction_Random(randomSoln,solutionSpace)
#        plt.figure()
#        plt.imshow(randomSoln)
        results[solnHash] = score
        resultsArray.append(score)
        
    return resultsArray
    
    
        
def compareToRandom():
    
    # Results
    GACount = []
    GAScore = []
    randomCount = []
    randomScore = []    
    GACounts = {}

    
    dimension = 8
    possibleSet = [0,1]
    populationSize = 100
    nGens = [2,4,6,8,10]
    pC = 0.8
    pMa = 0.5
    pMs = 0.8
    elitism = True
    shouldIPrint = False
    # GA
    for nGen in nGens:
        reporter = GA_roulette_epistatic(dimension,possibleSet,populationSize,nGen,pC,pMa,pMs,shouldIPrint,elitism)
        gacount = populationSize * nGen
        gascore = max(reporter.topScores)
        GACount.append(gacount)
        GAScore.append(gascore)
        if gascore in GACounts:
            GACounts[gascore] += 1
        else:
            GACounts[gascore] = 1
        
    # Random Search
    randomRuns = 1000000
    randomResultsArray = evaluateRandom_epistatic(randomRuns,8)
    
    
    
    results_200 = max(randomResultsArray[0:200])
    randomCount.append(200)
    randomScore.append(results_200)    
    
    results_400 = max(randomResultsArray[0:400])
    randomCount.append(400)
    randomScore.append(results_400)
    
    results_600 = max(randomResultsArray[0:600])
    randomCount.append(600)
    randomScore.append(results_600)
    
    results_800 = max(randomResultsArray[0:800])
    randomCount.append(800)
    randomScore.append(results_800)
    
    results_1000 = max(randomResultsArray[0:1000])
    randomCount.append(1000)
    randomScore.append(results_1000)
    
    results_10000 = max(randomResultsArray[0:10000])
    randomCount.append(10000)
    randomScore.append(results_10000)
    
    results_20000 = max(randomResultsArray[0:20000])
    randomCount.append(20000)
    randomScore.append(results_20000)
    
    
    results_50000 = max(randomResultsArray[0:50000])
    randomCount.append(50000)
    randomScore.append(results_50000)
    
    
    results_100000 = max(randomResultsArray[0:100000])
    randomCount.append(100000)
    randomScore.append(results_100000)
    
    results_200000 = max(randomResultsArray[0:200000])
    randomCount.append(200000)
    randomScore.append(results_200000)
    
    results_300000 = max(randomResultsArray[0:300000])
    randomCount.append(300000)
    randomScore.append(results_300000)
    
    results_400000 = max(randomResultsArray[0:400000])
    randomCount.append(400000)
    randomScore.append(results_400000)
    
    results_500000 = max(randomResultsArray[0:500000])
    randomCount.append(500000)
    randomScore.append(results_500000)
    
    results_750000 = max(randomResultsArray[0:750000])
    randomCount.append(750000)
    randomScore.append(results_750000)
        
    results_1000000 = max(randomResultsArray)
    randomCount.append(1000000)
    randomScore.append(results_1000000)

    
    GASizes = []
    for score in GAScore:
        scoreCount = GACounts[score]
        GASizes.append(50*scoreCount)
    
    # courtesy of http://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot    
    fig = plt.figure()
    fig.suptitle("Number of Evaluations and Quality\nFor Genetic Algorithm and Random Search\nAllocation-Optimization\npC=0.8,pMa=0.5,pMs=0.8,elitism=True",verticalalignment='baseline')
    ax1 = fig.add_subplot(111)
    ax1.set_xscale('log')
    ax1.scatter(GACount,GAScore,s=50,c='b',marker='s',label="Genetic Algorithm")
    plt.xlabel("Number of simulations (log scale)")
    plt.ylabel("Solutuion Quality (400=Max)")
    ax1.scatter(randomCount,randomScore,s=50,c='r',marker='o',label="Random Search")
    plt.legend(loc='best')
    
        
    return GACount,GAScore,randomCount,randomScore,GACounts,GASizes
    
    
def GA_NonEpistatic_Loop(dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint=False,elitism=False):
#    solutionSpace = np.array([[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]],[[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]])
#    solutionSpace = np.array([[[10,10,1,1],[10,10,1,1],[1,1,10,10],[1,1,10,10]],[[1,1,10,10],[1,1,10,10],[10,10,1,1],[10,10,1,1]]])  
#    print("Solution Space\n" + str(solutionSpace))
    solutionSpace = np.array([[[10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10],
                               [10,10,1,1,10,10,1,1],
                               [10,10,1,1,10,10,1,1],
                               [1,1,10,10,1,1,10,10],
                               [1,1,10,10,1,1,10,10]],

                      [[1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1],
                       [1,1,10,10,1,1,10,10],
                       [1,1,10,10,1,1,10,10],
                       [10,10,1,1,10,10,1,1],
                       [10,10,1,1,10,10,1,1]]])  
                       
    shouldIPrint = False
    reporter = GA_roulette(solutionSpace,dimension,possibleSet,populationSize,nGenerations,pC,pMa,pMs,shouldIPrint,elitism)
    return reporter
    
    
def GA_MC_nonEpistatic():
    w = open('GA_MC_nonEpistatic.csv','w')
    dimension = 8
    possibleSet = [0,1]
#    populationSizes = [10,20,50,100]
#    nGens = [1,2,5,10,50]
#    pCs = [.5,.6,.7,.8,.9,1]
#    pMas = [0.1,0.3,0.5,0.7,0.9]
#    pMss = [0.1,0.3,0.5,0.7,0.9]
    populationSizes = [10,20,50]
    nGens = [5,10]
    pCs = [0.5,0.7,0.9]
    pMas = [0.1,0.3,0.5,0.7]
    pMss = [0.1,0.3,0.5,0.7,0.9]
    elitisms = [True,False]
    mc = 2
    
    results = pd.DataFrame (np.zeros(0,dtype=[('populationSize','i4'),('nGen','i4'),('pC','i4'),('pMa','i4'),('pMs','i4'),('elitism','i4'),('bestScore','i4')]))
    w.write("populationSize,nGen,pC,pMa,pMs,elitism,bestScore\n")
    # courtesy of http://technicaltidbit.blogspot.com/2013/06/create-empty-dataframe-in-pandas.html
    
    test = 1
    nTestsRequired = len(populationSizes)*len(nGens)*len(pCs)*len(pMas)*len(pMss)*len(elitisms)*mc
    for populationSize in populationSizes:
        for nGen in nGens:
            for pC in pCs:
                for pMa in pMas:
                    for pMs in pMss:
                        for elitism in elitisms:
                            for i in range(0,mc):
                                shouldIPrint = False
                                reporter = GA_NonEpistatic_Loop(dimension,possibleSet,populationSize,nGen,pC,pMa,pMs,shouldIPrint,elitism)
                                bestScore = np.max(reporter.topScores)
#                                print("Test: " + str(test) + " of " + str(nTestsRequired) + ",BestScore: " + str(bestScore) + ", Params: popSize:" + str(populationSize)+",nGen:"+str(nGen)+",pC:"+str(pC)+",pMa:"+str(pMa)+",pMs:"+str(pMs)+",elitism:"+str(elitism))
                                results = results.append({'populationSize':populationSize,'nGen':nGen,'pC':pC,'pMa':pMa,'pMs':pMs,'elitism':elitism,'bestScore':bestScore},ignore_index=True)
                                w.write(str(populationSize)+","+str(nGen)+","+str(pC)+","+str(pMa)+","+str(pMs)+","+str(elitism)+","+str(bestScore)+"\n")
                                if test % 20 == 0:
                                    print("Test: " + str(test) + " of " + str(nTestsRequired))
                                test += 1
    results.to_csv("QTGA_MC.csv")
    import statsmodels.formula.api as sm
    result = sm.ols(formula="bestScore~populationSize+nGen+pC+pMa+pMs+elitism",data=results).fit()
    print result.summary()
    w.close()
    return results