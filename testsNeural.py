from NeuralNetwork import *

print("TESTING SIGMOID")
print(sigmoid(-np.inf))
print(sigmoid(-1))
print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(np.inf))

print("\nTESTING CLASS")
neural1 = NeuralNetwork([4,5,3])
print("TESTING LAYER SCHEMA")
print(neural1.layerSchema)
print("INTER LAYER MATRICES")
print(neural1.transforms)
neural2 = NeuralNetwork([[[6,3,1]],[[1,0,1],[2,2,2],[1,2,3]]])
print("TESTING LAYER SCHEMA")
print(neural2.layerSchema)
print("INTER LAYER MATRICES")
print(neural2.transforms)

print("TESTING EVALUATION")
print(neural1.evaluate([1,2,3,4]))

print("TESTING EVALUATION")
print(neural2.evaluate([1]))

print("FAILING CROSSOVER")
neuralf = neural1 + neural2
print(neuralf)
neural3 = NeuralNetwork([4,5,3])

neuralChild1, neuralChild2 = neural1 + neural3
print(neuralChild1.transforms)
print(neuralChild2.transforms)
