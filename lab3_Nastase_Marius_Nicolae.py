import numpy as np
import matplotlib.pyplot as plt


def FFT(x):
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
            np.exp(-2j * np.pi * np.arange(N) / N)

        X = np.concatenate( \
            [X_even + factor[:int(N / 2)] * X_odd,
             X_even + factor[int(N / 2):] * X_odd])
        return X

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

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.real(F), cmap='hot', interpolation='nearest')
    plt.title("Partea Reala a Matricei Fourier")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.imag(F), cmap='hot', interpolation='nearest')
    plt.title("Partea Imaginara a Matricei fourier")
    plt.colorbar()

    plt.show()

def ex2():
    frecventa_semnal = 2.0 # diferita fata de cea data
    N = 100 # numarul de esantioane

    # semnalul sinusoidal
    timp = np.arange(N)
    semnal = np.sin(2 * np.pi * frecventa_semnal * timp / N)

    # Calculati si afisati infasurarea pe cercul unitate pentru Figura1
    gama = np.linspace(0,1,N)
    for i in range(N):
        x = semnal[i] * np.exp(-2 * np.pi * gama[i] * 1j * timp)
        plt.polar(np.angle(x), np.abs(x), marker = 'o', color = plt.cm.viridis(i / N))
    plt.title("Figura 1: Infasurarea semnalului pe cercul unitate")
    plt.show()

    gama_values = [0.25,0.5,0.75,frecventa_semnal]
    plt.figure(figsize=(12,6))

    for gama_ in gama_values:
        x = semnal * np.exp(-2 * np.pi * gama_ * 1j * timp)
        plt.plot(timp, np.abs(x), label=f"Gama = {gama_}")
    plt.title("Figura 2: Infasurarea semnalului pentru diferite valori ale lui Gama")
    plt.xlabel("n")
    plt.ylabel("Amplitudine")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # ex1()
    ex2()
