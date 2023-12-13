import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


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


def inmultire_polinoame_directa(p, q):
    # Gradul maxim al rezultatului
    putere_maxima = len(p) + len(q) - 2

    # Initializare rezultat cu coeficienti 0
    r = np.zeros(putere_maxima + 1, dtype=int)

    # Inmultirea polinoamelor
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]

    return r


def inmultire_polinoame_fft(p, q):
    # Gradul maxim al rezultatului
    putere_maxima = len(p) + len(q) - 2

    # Adaugare zerouri la polinoame pentru a se potrivi cu gradul rezultatului
    p_padded = np.pad(p, (0, putere_maxima - len(p) + 1), mode='constant')
    q_padded = np.pad(q, (0, putere_maxima - len(q) + 1), mode='constant')

    # Transformata Fourier rapida a celor doua polinoame
    P = fft(p_padded)
    Q = fft(q_padded)

    # Inmultirea in domeniul Fourier
    R = P * Q

    # Transformata Fourier inversa pentru a obtine rezultatul
    result_fft = np.real(ifft(R))

    # Rotunjire la intreg pentru a evita erori de virgula mobila
    result_fft = np.round(result_fft).astype(int)

    return result_fft
def ex2():
    p = np.random.randint(-10, 10, size=5)
    q = np.random.randint(-10, 10, size=4)

    # Calculare produs direct
    result_direct = inmultire_polinoame_directa(p, q)
    print("Produsul folosind inmultirea directa:", result_direct)

    # Calculare produs folosind FFT
    result_fft = inmultire_polinoame_fft(p, q)
    print("Produsul folosind FFT:", result_fft)

def fereastra_dreptunghiulara(Nw):
    return np.ones(Nw)

def fereastra_Hanning(Nw):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(Nw) / (Nw - 1)))

def plot_fereastra_si_semnal(fereastra, Nw, semnal, titlu):
    plt.figure(figsize=(10, 4))

    # Afisare fereastra
    plt.subplot(121)
    plt.plot(fereastra)
    plt.title(f'Fereastra {titlu}')

    # Afisare sinusoida trecuta prin fereastra
    plt.subplot(122)
    plt.plot(semnal)
    plt.title(f'Sinusoida trecuta prin fereastra {titlu}')

    plt.tight_layout()
    plt.show()
def ex3():
    Nw = 200
    f = 100
    A = 1
    phi = 0
    t = np.arange(Nw) / A

    # Construirea sinusoidei
    signal = A * np.sin(2 * np.pi * f * t + phi)

    # Construirea ferestrelor
    rectangular = fereastra_dreptunghiulara(Nw)
    hanning = fereastra_Hanning(Nw)

    # Afisarea graficelor
    plot_fereastra_si_semnal(rectangular, Nw, signal, 'Dreptunghiulară')
    plot_fereastra_si_semnal(hanning, Nw, signal, 'Hanning')


if __name__ == "__main__":
    # ex1()
    # ex2()
    ex3()