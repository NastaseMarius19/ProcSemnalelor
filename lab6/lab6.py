import numpy as np
import matplotlib.pyplot as plt
def ex1():
    N = 100
    x = np.random.rand(N)

    # Afisare grafic initial
    plt.figure(figsize=(12, 3))
    plt.subplot(141)
    plt.plot(x)
    plt.title('Graficul initial')

    # Iteratie x ← x * x de trei ori
    for i in range(3):
        x = x * x

        # Afisare grafic pentru fiecare iteratie
        plt.subplot(142 + i)
        plt.plot(x)
        plt.title(f'Iteratia {i + 1}')

    # Afisare toate graficele
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ex1()
