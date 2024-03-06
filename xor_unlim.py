# import numpy as np
# import pprint

# # Определение функции активации (сигмоидной)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Определение производной функции активации (сигмоидной)
# def sigmoid_derivative(x):
#     return x * (1 - x)

# class Layer:
#     def __init__(self, input_size, output_size):
#         self.weights = np.random.rand(input_size, output_size)
#         self.bias = np.random.rand(output_size)

#     def forward(self, inputs):
#         self.inputs = inputs
#         self.outputs = sigmoid(np.dot(inputs, self.weights) + self.bias)

#     def backward(self, prev_delta, learning_rate):
#         # Находим производную
#         derivative = prev_delta * sigmoid_derivative(self.outputs)
#         weights_grads = np.dot(self.inputs.T, derivative)
#         bias_grads = np.sum(derivative)
#         delta = np.dot(prev_delta, self.weights.T)

#         self.weights += learning_rate * weights_grads
#         self.bias += learning_rate * bias_grads

#         return delta

# class NeuralNetwork:
#     def __init__(self, layers):
#         self.layers = []
#         for neurons in range(len(layers) - 1):
#             layer = Layer(layers[neurons], layers[neurons + 1])
#             self.layers.append(layer)

#     def forward(self, inputs):
#         for layer in self.layers:
#             layer.forward(inputs)
#             inputs = layer.outputs

#     def backward(self, delta, learning_rate):
#         for layer in reversed(self.layers):
#             delta = layer.backward(delta, learning_rate)

#     def train(self, X, Y, epochs, learning_rate):
#         for epoch in range(epochs):
#             self.forward(X)
        
#             output_error = Y - self.layers[-1].outputs
#             output_delta = output_error * sigmoid_derivative(self.layers[-1].outputs)

#             self.backward(output_delta, learning_rate)
            
#             if epoch % (epochs // 10) == 0:
#                 loss = np.mean(np.abs(output_error))
#                 print(f"Epoch: {epoch}, Loss: {loss}")
        
#     def predict(self, X):
#         self.forward(X)
#         return self.layers[-1].outputs.round()

# network = NeuralNetwork([2, 4, 4, 1])

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y = np.array([[1], [0], [0], [1]])

# network.train(X, Y, 20000, 0.1)
# pprint.pprint(network.predict(X))

import numpy as np
import pprint
import matplotlib.pyplot as plt

# Определение функции активации (сигмоидной)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Определение производной функции активации (сигмоидной)
def sigmoid_derivative(x):
    return x * (1 - x)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = sigmoid(np.dot(inputs, self.weights) + self.bias)

    def backward(self, prev_delta, learning_rate):
        # Находим производную
        derivative = prev_delta * sigmoid_derivative(self.outputs)
        weights_grads = np.dot(self.inputs.T, derivative)
        bias_grads = np.sum(derivative)
        delta = np.dot(prev_delta, self.weights.T)

        self.weights += learning_rate * weights_grads
        self.bias += learning_rate * bias_grads

        return delta

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = []
        for neurons in range(len(layers) - 1):
            layer = Layer(layers[neurons], layers[neurons + 1])
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs

    def backward(self, delta, learning_rate):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
        
            output_error = Y - self.layers[-1].outputs
            output_delta = output_error * sigmoid_derivative(self.layers[-1].outputs)

            self.backward(output_delta, learning_rate)

            return output_error
        
    def predict(self, X):
        self.forward(X)
        return self.layers[-1].outputs.round()

network = NeuralNetwork([2, 4, 4, 1])
losses= []

for epoch in range(20000):
    X = np.array([[0, 0]])
    Y = np.array([[0]])
    network.train(X, Y, 1, 0.1)

    X = np.array([[0, 1]])
    Y = np.array([[1]])
    network.train(X, Y, 1, 0.1)

    X = np.array([[1, 0]])
    Y = np.array([[1]])
    network.train(X, Y, 1, 0.1)

    X = np.array([[1, 1]])
    Y = np.array([[0]])
    output_error = network.train(X, Y, 1, 0.1)

    if epoch % (20000 // 10) == 0:
        loss = np.mean(np.abs(output_error))
        print(f"Epoch: {epoch}, Loss: {loss}")
        losses.append(loss) 


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#Y = np.array([[0], [1], [1], [0]])
#network.train(X, Y, 20000, 0.1)
pprint.pprint(network.predict(X))

plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show()