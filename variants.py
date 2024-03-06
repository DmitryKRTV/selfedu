# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Входные данные XOR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# # Ожидаемые значения XOR
# y = np.array([[0], [1], [1], [0]])

# # Инициализация весов случайными значениями
# np.random.seed(0)
# W1 = 2 * np.random.random((2, 2)) - 1
# W2 = 2 * np.random.random((2, 1)) - 1

# # Функция активации (сигмоид)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Процесс обучения
# epochs = 60000
# learning_rate = 0.1
# losses = []  # Список для сохранения значений функции потерь

# for epoch in range(epochs):
#     # Прямое распространение (forward pass)
#     hidden_layer = sigmoid(np.dot(X, W1))
#     output_layer = sigmoid(np.dot(hidden_layer, W2))
    
#     # Ошибка
#     error = y - output_layer
#     loss = np.mean(np.abs(error))  # Функция потерь (средняя абсолютная ошибка)
#     losses.append(loss)
    
#     # Обратное распространение (backpropagation)
#     output_delta = error * (output_layer * (1 - output_layer))
#     hidden_delta = output_delta.dot(W2.T) * (hidden_layer * (1 - hidden_layer))
    
#     # Обновление весов
#     W2 += hidden_layer.T.dot(output_delta) * learning_rate
#     W1 += X.T.dot(hidden_delta) * learning_rate

# # Тестирование модели
# test_input = np.array([[1, 1]])
# predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, W1)), W2))


# print("Predicted Output:")
# print(predicted_output)


# # Визуализация графика функции потерь
# plt.plot(losses)

# plt.xlabel('Эпохи')
# plt.ylabel('Функция потерь')
# plt.title('График функции потерь в процессе обучения')
# plt.show()

# # Визуализация трехмерного графика
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], y.ravel(), c='b', marker='o')  # Точки данных
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Выход')

# # Создание сетки точек для предсказания
# x1 = np.linspace(0, 1, 10)
# x2 = np.linspace(0, 1, 10)
# X1, X2 = np.meshgrid(x1, x2)
# X_grid = np.c_[X1.ravel(), X2.ravel()]

# # Вычисление предсказаний для сетки точек
# hidden_layer_grid = sigmoid(np.dot(X_grid, W1))
# y_grid = sigmoid(np.dot(hidden_layer_grid, W2))

# # Построение поверхности
# ax.plot_surface(X1, X2, y_grid.reshape(X1.shape), alpha=0.5)

# plt.title('Разделение результатов на плоскости с помощью весов')
# plt.show()

# print("Predicted Output:")
# print(predicted_output)

# import numpy as np
# import matplotlib.pyplot as plt

# # Входные данные XOR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# # Ожидаемые значения XOR
# y = np.array([[0], [1], [1], [0]])

# # Инициализация весов случайными значениями
# np.random.seed(0)
# W1 = 2 * np.random.random((2, 2)) - 1
# W2 = 2 * np.random.random((2, 1)) - 1

# # Функция активации (сигмоид)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Процесс обучения
# epochs = 100000
# learning_rate = 0.1
# losses = []  # Список для сохранения значений функции потерь

# for epoch in range(epochs):
#     # Прямое распространение (forward pass)
#     hidden_layer = sigmoid(np.dot(X, W1))
#     output_layer = sigmoid(np.dot(hidden_layer, W2))
    
#     # Ошибка
#     error = y - output_layer
#     loss = np.mean(np.abs(error))  # Функция потерь (средняя абсолютная ошибка)
#     losses.append(loss)
    
#     # Обратное распространение (backpropagation)
#     output_delta = error * (output_layer * (1 - output_layer))
#     hidden_delta = output_delta.dot(W2.T) * (hidden_layer * (1 - hidden_layer))
    
#     # Обновление весов
#     W2 += hidden_layer.T.dot(output_delta) * learning_rate
#     W1 += X.T.dot(hidden_delta) * learning_rate

# # Тестирование модели
# test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, W1)), W2))

# print("Predicted Output:")
# print(predicted_output)

# # Визуализация классификации с помощью весов
# x_min, x_max = 0, 1
# y_min, y_max = 0, 1
# h = 0.01  # Шаг сетки
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = sigmoid(np.dot(sigmoid(np.c_[xx.ravel(), yy.ravel()]), W1)).dot(W2)
# Z = np.round(Z).reshape(xx.shape)

# plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors='k')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Влияние весов на классификацию')
# plt.show()

# ---------------------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt

# # Входные данные XOR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# # Ожидаемые значения XOR
# y = np.array([[0], [1], [1], [0]])

# # Инициализация весов случайными значениями
# np.random.seed(0)
# W1 = 2 * np.random.random((2, 2)) - 1
# W2 = 2 * np.random.random((2, 1)) - 1

# # Функция активации (сигмоид)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Процесс обучения
# epochs = 10000
# learning_rate = 0.1
# losses = []  # Список для сохранения значений функции потерь

# for epoch in range(epochs):
#     # Прямое распространение (forward pass)
#     hidden_layer = sigmoid(np.dot(X, W1))
#     output_layer = sigmoid(np.dot(hidden_layer, W2))
    
#     # Ошибка
#     error = y - output_layer
#     loss = np.mean(np.abs(error))  # Функция потерь (средняя абсолютная ошибка)
#     losses.append(loss)
    
#     # Обратное распространение (backpropagation)
#     output_delta = error * (output_layer * (1 - output_layer))
#     hidden_delta = output_delta.dot(W2.T) * (hidden_layer * (1 - hidden_layer))
    
#     # Обновление весов
#     W2 += hidden_layer.T.dot(output_delta) * learning_rate
#     W1 += X.T.dot(hidden_delta) * learning_rate

# # Тестирование модели
# test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, W1)), W2))

# # Визуализация классификации с помощью весов
# x_min, x_max = np.min(W1), np.max(W1)
# y_min, y_max = np.min(W2), np.max(W2)
# h = 0.01  # Шаг сетки
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# grid = np.c_[xx.ravel(), yy.ravel()]
# hidden_layer_grid = sigmoid(np.dot(grid, W1))
# Z = sigmoid(np.dot(hidden_layer_grid, W2))
# Z = np.round(Z).reshape(xx.shape)

# plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8, levels=1)
# plt.scatter(W1[:, 0], W1[:, 1], c='black', cmap=plt.cm.RdBu, edgecolors='k')
# plt.xlabel('Weight 1')
# plt.ylabel('Weight 2')
# plt.title('Влияние весов на классификацию')
# plt.show()

# print("Predicted Output:")
# print(predicted_output)

# ---------------------------------------------------------------------------------------


# import numpy as np
# import matplotlib.pyplot as plt

# # Диапазон значений x
# x = np.linspace(-20, 10, 100)
# print(x)

# # Функция f(x) = 2/(1 + np.exp(-x)) -
# y = 2 / (1 + np.exp(-x)) - 1

# # Построение графика
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('График функции f(x) = 2/(1 + np.exp(-x)) -')
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Задаем диапазон значений для переменной x
# x = np.linspace(-10, 10, 100)

# # Задаем значения весов с шагом 0.1 от -0.5 до 0.4
# weights = np.arange(-0.5, 0.5, 0.1)

# # Создаем графики
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
# axes = axes.flatten()

# # Проходим по каждому значению веса и строим соответствующий график
# for i, weight in enumerate(weights):
#     # Вычисляем линейную комбинацию веса и переменной x
#     z = weight * x

#     # Применяем сигмоидную функцию к линейной комбинации
#     output = sigmoid(z)

#     # Строим график
#     axes[i].plot(x, output)
#     axes[i].set_title(f'Weight: {weight}')
#     axes[i].set_xlabel('x')
#     axes[i].set_ylabel('sigmoid(x)')

# # Убираем лишние подзаголовки
# for j in range(len(weights), len(axes)):
#     fig.delaxes(axes[j])

# # Размещаем графики
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Определение функции активации (сигмоидной)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Определение функции потерь (среднеквадратичная ошибка)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Определение класса для нейросети
class XORNeuralNetwork:
    def __init__(self):
        # Инициализация весов случайными значениями
        self.weights1 = np.random.rand(2, 2)
        self.bias1 = np.random.rand(2)
        self.weights2 = np.random.rand(2, 1)
        self.bias2 = np.random.rand(1)
        # Создание списков для хранения истории прогресса классификации
        self.loss_history = []
        self.epoch_history = []

    def forward(self, X):
        # Прямое распространение
        self.hidden_layer = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)

    def backward(self, X, y, learning_rate):
        # Обратное распространение
        output_error = y - self.output
        output_delta = output_error * (self.output * (1 - self.output))
       
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * (self.hidden_layer * (1 - self.hidden_layer))

        self.weights2 += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.bias2 += learning_rate * np.sum(output_delta)

        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
        self.bias1 += learning_rate * np.sum(hidden_delta)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Прямое и обратное распространение для каждой эпохи
            self.forward(X)
            self.backward(X, y, learning_rate)
            # Вычисление функции потерь и добавление в историю
            loss = mse_loss(y, self.output)
            self.loss_history.append(loss)
            self.epoch_history.append(epoch)

            # Отображение классификации и разделяющей границы для каждой 10-й эпохи
            if epoch % (epochs // 10) == 0:
                if epoch == 0:
                    plt.figure(figsize=(15, 6))
                plt.subplot(2, 5, epoch // (epochs // 10) + 1)
                self.plot_classification(X, y, epoch, loss)
                self.plot_decision_boundary(X, y)

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        self.forward(X)
        return self.output.round()

    def plot_classification(self, X, y, epoch, loss):
        class0 = y.flatten() == 0
        class1 = y.flatten() == 1
        plt.scatter(X[class0, 0], X[class0, 1], c='blue', label='Class 0')
        plt.scatter(X[class1, 0], X[class1, 1], c='red', label='Class 1')
        plt.title(f'Epoch: {epoch}, Loss: {loss:.4f}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()

    def plot_decision_boundary(self, X, y):
        # Генерация равномерной сетки для построения разделяющей границы
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Прогнозирование классификации на сетке
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        # Отображение разделяющей границы
        plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.5)
        plt.colorbar()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

# Генерация входных данных XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [0], [0], [1]])

# Создание и обучение нейросети
network = XORNeuralNetwork()
network.train(X, y, epochs=10000, learning_rate=0.1)

# Предсказание и отображение классификации после обучения
predictions = network.predict(X)
print("Predictions:")
print(predictions)