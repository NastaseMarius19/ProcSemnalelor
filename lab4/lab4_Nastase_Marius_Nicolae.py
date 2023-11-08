import numpy as np
import time
import matplotlib.pyplot as plt


def ex1():
    N = [128, 256, 512, 1024, 2048, 4096, 8192]
    manual_times = []
    numpy_times = []

    for n in N:
        # Generarea unui semnal de test
        semnal = np.random.rand(n) + 1j * np.random.rand(n)

        # Calcul timp executie pentru Transformata Fourier Discreta implementata manual
        start_time = time.time()
        semnal_manual = np.zeros(n, dtype=complex)
        for k in range(n):
            for m in range(n):
                semnal_manual[k] += semnal[m] * np.exp(-2j * np.pi * k * m / n)
        manual_times.append(time.time() - start_time)

        # Calcul timp executie pentru np.fft.fft
        start_time = time.time()
        semnal_dft_numpy = np.fft.fft(semnal)
        numpy_times.append(time.time() - start_time)

    # Creare și afișare grafic
    plt.figure(figsize=(8, 6))
    plt.plot(N, manual_times, label='DFT Manual', marker='o')
    plt.plot(N, numpy_times, label='np.fft.fft', marker='x')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Dimensiunea Vectorului (N)')
    plt.ylabel('Timp de Execuție (s)')
    plt.legend()
    plt.grid()
    plt.title('Compararea timpului de execuție pentru DFT Manual și np.fft.fft')
    plt.show()

def ex2_3(frecventa_sub_Nyquist):
    # Parametrii semnalului sinusoidal
    frecventa = 5
    amplitudine = 1.0
    faza = 0

    perioada = 1.0 / frecventa
    perioada_esantionare = 1.0 / frecventa_sub_Nyquist

    # Vectorul timpului
    t_original = np.linspace(0, 4 * perioada, 1000)

    # Generarea semnalului sinusoidal
    semnal_original = amplitudine * np.sin(2 * np.pi * frecventa * t_original + faza)

    t_esantionat = np.arange(0, t_original[-1], perioada_esantionare)

    semnal_esantionat = amplitudine * np.sin(2 * np.pi * frecventa * t_esantionat + faza)

    # Afișarea semnalului original și a semnalului esantionat
    plt.figure(figsize=(12, 6))

    plt.subplot(2,1,1)
    plt.plot(t_original, semnal_original)
    plt.title('Semnalul Original')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')

    plt.subplot(2,1,2)
    plt.plot(t_original, semnal_original, label='Semnal Original', color='b')
    plt.stem(t_esantionat, semnal_esantionat, basefmt='b-', linefmt='r-', markerfmt='ro')
    plt.title('Semnal Esantionat cu Frecventa Sub-Nyquist')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')

    plt.tight_layout()
    plt.show()

def ex4():
    # Frecventele emise de un contrabas se incadreaza intre 40Hz si 200Hz => frecvența maximă din spectrul semnalului original este de 200 Hz
    # Frecventa Nyquist este jumătate din frecventa de esantionare maximă posibilă.
    # Deci, pentru a acoperi toate frecventele posibile ale semnalului de contrabas,
    # frecvența de esantionare minimă trebuie să fie de cel puțin dublul acestei
    # frecvențe maxime, adică 2 * 200 Hz = 400 Hz.
    print("Frecventa minima este de: ", 400)

def ex5():
    """
    Ați menționat că P_semnal_dB este de 90 dB și SNR_dB este de 80 dB. Pentru a calcula puterea zgomotului (P_zgomot), folosim formula:

    P_zgomot_dB = P_semnal_dB - SNR_dB

    Pentru a găsi puterea zgomotului în decibeli, putem folosi această formulă:

    P_zgomot_dB = 90 dB - 80 dB = 10 dB

    Deci, puterea zgomotului este de 10 dB.
    """
    print("puterea zgomotului este de 10 dB")

if __name__ == "__main__":
    #ex1()
    #ex2_3(6)
    #ex2_3(12)
    #ex4()
    ex5()
