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

def ex3():
    import numpy as np
    import matplotlib.pyplot as plt

    N = 1000
    t = np.linspace(0, 1, N)

    f = [20, 30, 50]  # Frecventele semnalelor (4 componente)

    semnal = np.zeros_like(t)  # Semnalul compus
    for fi in f:
        semnal += np.sin(2 * np.pi * fi * t)

    semnal_dft = np.zeros(N, dtype=complex)  # Transformata Fourier
    for k in range(N):
        for n in range(N):
            semnal_dft[k] += semnal[n] * np.e ** (-2j * np.pi * k * n / N)

    semnal_dft = np.abs(semnal_dft)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(t, semnal)
    axs[0].set_title("Semnalul x(t)")
    axs[0].set_xlabel("Timp")
    axs[0].set_ylabel("Amplitudine")
    axs[0].grid()

    axs[1].stem(np.linspace(0, 100, 100), semnal_dft[:100])
    axs[1].set_title("Transformata Fourier a semnalului")
    axs[1].set_xlabel("Frecventa")
    axs[1].set_ylabel("Amplitudine")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #ex1()
    #ex2()
    ex3()
