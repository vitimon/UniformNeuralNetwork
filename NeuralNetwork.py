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

    def __init__(self, layers):
        self.layerSchema = layers
        if len(layers) < 2: return("small number of layers")
        lastSize = layers[0]
        self.transforms = []
        for layerSize in layers[1:]:
            self.transforms += [generateMatrix(lastSize, layerSize)]
            lastSize = layerSize

    def __init__(self, transforms):
        self.transforms = transforms
        schema = []
        schema += len(transforms[0])
        for transform in transforms:
            schema += len(transform[0])
        self.layerSchema = schema

            
    def evaluate(self, inputs):
        if self.layerSchema[0] != len(inputs): return("input doesnt match neural network")
        inputLayer = normalize(inputs)
        for transform in self.transforms:
            nextLayer = inputLayer.dot(transform)
            nextLayer = normalize(inputLayer)
            inputLayer = nextLayer
        return nextLayer

    #using dunder sum to try crossover networks
    def __add__(self,other):
        if self.layerSchema != other.layerSchema: return("Networks doesnt match")
        cross1, cross2 = [], []
        for index in range(len(self.transforms)):
            parent1, parent2 = self.transforms[index], other.transforms[index]
            from1to2 = np.linalg.inv(parent1).dot(parent2)**(1/3)
            cross1 += parent1.dot(from1to2)
            cross2 += parent1.dot(from1to2**2)
        return NeuralNetwork(cross1), NeuralNetwork(cross2)