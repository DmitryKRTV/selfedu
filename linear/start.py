import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 5
    b = 3

    x1 = np.random.random(N)
    x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b

    C1 = [x1, x2]

    x1 = np.random.random(N)
    x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 + b

    C2 = [x1, x2]
    print(C2)

    f = [0 + b, 1 + b]
    
    w2 = 0.5
    w3 = -b * w2
    w = np.array([-w2, w2, w3])

    for i in range(N):
        x = np.array([C2[0][i], C2[1][i], 1])
        y = np.dot(w, x)
        if y >= 0:
            print("C1 dot")
        else: 
            print("C2 dot")

    plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
    plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
    plt.plot(f)
    plt.grid(True)
    plt.show()

#Объяснение нейрона смещения это конечно херня из под коня полная
# "Автор заяявляет: А давайте предположим, что значения смещены. И как нам их классифицировать?"
# "Абсурд в том, что он нихрена не говорит почему они смещены, а берёт и смещает относительно этого же созданного биаса."
# "То есть то, что это сделал интерпритируется как: А давайте создадим проблему просто так и выбъем клин(проблему смещения) нейроном клином(смещением, которое и создало эту проблему)"
# "Занавес"


if __name__ == "__main__":
    main()