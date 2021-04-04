#Juan Horacio Rivera Peña
import numpy as np

def sigmoid(x):

    return 1 / (1 + np.exp(-x))


bias = 2
vector_input =np.array([1.72,1.23,bias])
weights = np.array([1.26,2.17,1])

first_layer = np.dot(vector_input, weights)

prediction = sigmoid(first_layer)

print("prediction = {}".format(prediction))
