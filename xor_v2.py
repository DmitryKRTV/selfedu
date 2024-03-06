import numpy as np

# Определение функции активации (сигмоидной)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Определение производной функции активации (сигмоидной)
def sigmoid_derivative(x):
    return x * (1 - x)

# Класс, представляющий слой нейронов
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(inputs, self.weights) + self.bias)
    
    def backward(self, output_delta, learning_rate):
        weight_gradients = np.dot(self.inputs.T, output_delta * sigmoid_derivative(self.output))
        bias_gradients = np.sum(output_delta * sigmoid_derivative(self.output), axis=0)
        input_delta = np.dot(output_delta, self.weights.T)
        
        self.weights += learning_rate * weight_gradients
        self.bias += learning_rate * bias_gradients
        
        return input_delta

# Класс, представляющий нейронную сеть
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
    
    def backward(self, output_delta, learning_rate):
        input_delta = output_delta
        for layer in reversed(self.layers):
            input_delta = layer.backward(input_delta, learning_rate)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            
            # Вычисление ошибки и дельты для обратного распространения
            output_error = y - self.layers[-1].output
            output_delta = output_error * sigmoid_derivative(self.layers[-1].output)
            
            self.backward(output_delta, learning_rate)
            
            if epoch % (epochs // 10) == 0:
                loss = np.mean(np.abs(output_error))
                print(f"Epoch: {epoch}, Loss: {loss}")
    
    def predict(self, X):
        self.forward(X)
        return self.layers[-1].output.round()

# Генерация входных данных XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [0], [0], [1]])

# Создание и обучение нейронной сети
network = NeuralNetwork([2, 4, 4, 1])
network.train(X, y, epochs=20000, learning_rate=0.1)

# Прогнозирование
predictions = network.predict(X)
print("Predictions:", predictions)



