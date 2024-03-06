import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(2, 2)
        self.weights2 = np.random.rand(2, 1)
        self.bias1 = np.random.rand(2)
        self.bias2 = np.random.rand(1)

    def forward(self, X):
        self.H = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.O = sigmoid(np.dot(self.H, self.weights2) + self.bias2)

    def backward(self, X, Y, lr):
        self.O_error = Y - self.O
        self.O_delta = self.O_error * (self.O * (1 - self.O))

        self.H_error = np.dot(self.O_delta, self.weights2.T)
        self.H_delta = self.H_error * (self.H * (1 - self.H))

        self.weights2 += lr * np.dot(self.H.T, self.O_delta)
        self.bias2 += lr * np.sum(self.O_delta)

        self.weights1 += lr * np.dot(X.T, self.H_delta)
        self.bias1 += lr * np.sum(self.H_delta)

    def predict(self, X):
        self.forward(X)
        return self.O.round()

    def log(self):
        print('---weights1---')
        print(f'{self.weights1}')
        print('---weights2---')
        print(f'{self.weights2}')
        print('---bias1---')
        print(f'{self.bias1}')
        print('---bias2---')
        print(f'{self.bias2}')
        print('---H---')
        print(f'{self.H}')
        print('---O---')
        print(f'{self.O}')


network = NeuralNetwork(2, )
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

for i in range(10000):
    network.forward(X)
    network.backward(X, Y, 0.1)

predict = network.predict(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))

network.log()

print(predict)