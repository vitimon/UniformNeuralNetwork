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

class NeuralNetwork:

    def normalize(layer, inputs):
        layer = list(map(sigmoid, inputs))
        return np.array(layer) 

    def __init__(self, layers):
        self.layerSchema = layers
        if len(layers) < 2: raise("small number o layers")
        lastSize = layers[0]
        self.transforms = []
        for layerSize in layers[1:]:
            self.transforms += [generateMatrix(lastSize, layerSize)]
            lastSize = layerSize
            
    def evaluate(self, inputs):
        if self.layerSchema[0] != len(inputs): raise("input doesnt match neural network")
        nextLayer = list(map(sigmoid, inputs))
        nextLayer = np.array(nextLayer)
        for transform in self.transforms:
            nextLayer = nextLayer.dot(transform)
            nextLayer = list(map(sigmoid, nextLayer))
            nextLayer = np.array(nextLayer)
        return nextLayer