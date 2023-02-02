import numpy as np
from random import random

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def randomElement():
    return random() - random()

def generateMatrix(lines,columns):
    matrix = []
    for line in range(lines):
        lineArray = []
        for column in range(columns):
            lineArray += [randomElement()]
        matrix += [lineArray]
    return np.array(matrix)

def normalize(inputs):        
        return np.array(list(map(sigmoid, inputs)))

class NeuralNetwork:

    def __initiateFromLayers(self, layers: list[int]) -> None:
        print("LAYnit")
        self.layerSchema = layers
        if len(layers) < 2: return("small number of layers")
        lastSize = layers[0]
        self.transforms = []
        for layerSize in layers[1:]:
            self.transforms += [generateMatrix(lastSize, layerSize)]
            lastSize = layerSize

    def __initiateFromTransforms(self, transforms: list[np.array]) -> None:
        print("TRAnit")
        self.transforms = transforms
        schema = []
        schema += [len(transforms[0])]
        for transform in transforms:
            schema += [len(transform[0])]
        self.layerSchema = schema

    def __init__(self, arg):
        if isinstance(arg[0], int): self.__initiateFromLayers(arg)
        elif isinstance(arg[0], list): self.__initiateFromTransforms(arg)

            
    def evaluate(self, inputs):
        if self.layerSchema[0] != len(inputs): return("input doesnt match neural network")
        inputLayer = normalize(inputs)
        for transform in self.transforms:
            nextLayer = normalize(inputLayer.dot(transform))
            inputLayer = nextLayer
        return nextLayer

    #using dunder sum to try crossover networks
    def __add__(self,other):
        if self.layerSchema != other.layerSchema: return("Networks doesnt match")
        cross1, cross2 = [], []
        for index in range(len(self.transforms)):
            parent1, parent2 = self.transforms[index], other.transforms[index]
            inverted = np.linalg.pinv(parent1)
            from1to2 = ((inverted).dot(parent2))
            step = from1to2**(1/3)
            cross1 += [parent1.dot(step)]
            cross2 += [parent1.dot(step**2)]
        return NeuralNetwork(cross1), NeuralNetwork(cross2)