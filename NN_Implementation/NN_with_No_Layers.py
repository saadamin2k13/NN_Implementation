import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_dervative(x):
    return x * (1-x)


training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

weight = 2 * np.random.random((3,1)) - 1

print('Random starting weights')
print(weight)

for iteration in range(100000):
    input_layer = training_inputs
    output = sigmoid(np.dot(input_layer, weight))

    error = training_outputs - output

    adjustments = error * sigmoid_dervative(output)

    weight += np.dot(input_layer.T, adjustments)

print('weights after training')
print(weight)

print('output after training')
print(output)

