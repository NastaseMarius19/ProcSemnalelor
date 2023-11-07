import numpy as np
import matplotlib.pyplot as plt

def ex1():
    N = 8

    F = np.zeros((N, N), dtype=complex)

    # construirea matricei Fourier
    for i in range(N):
        for j in range(N):
            F[i, j] = np.exp(-2j * np.pi * i * j / N) / np.sqrt(N)

    # Verificam unitaritatea
    F_H = np.conjugate(F).T  # transpusa conjugata
    identity_matrix = np.identity(N)

    is_unitary = np.allclose(np.dot(F_H, F), N * identity_matrix)

    print("Este matricea Fourier unitara: ", is_unitary)

    # Afisare grafice pentru fiecare componenta de frecventa
    plt.figure(figsize=(12, 8))

    for k in range(N):
        plt.subplot(N, 2, 2 * k + 1)
        plt.plot(np.real(F[:, k]))
        plt.title(f"Partea Reala a Componentei {k + 1}")
        plt.subplot(N, 2, 2 * k + 2)
        plt.plot(np.imag(F[:, k]))
        plt.title(f"Partea Imaginara a Componentei {k + 1}")
        plt.tight_layout()

    plt.show()

def ex2():
    frecventa_semnal = 5  # Frecventa semnalului
    frecventa_esantionare = 100  # Frecventa de esantionare

    timp = np.linspace(0, 1, frecventa_esantionare)  # Timpul semnalului
    x = np.sin(2 * np.pi * frecventa_semnal * timp)

    # Infasurarea semnalului x(t) pe cercul unitate
    y = x * np.exp(-2j * np.pi * timp)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(timp, x)
    axs[0].set_title("Semnalul X")
    axs[0].set_xlabel("Timp")
    axs[0].set_ylabel("Amplitudine")
    axs[0].grid()

    axs[1].plot(y.real, y.imag, c='green')
    axs[1].set_title("Reprezentarea semnalului in planul complex (infasurarea)")
    axs[1].set_xlabel("Real")
    axs[1].set_ylabel("Imaginar")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #ex1()
    ex2()
