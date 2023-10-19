
# Laborator 2 Procesare Semnalelor - Nastase Marius Nicolae - grupa 461
import numpy as np
import matplotlib.pyplot as plt

timp = np.linspace(0, 2 * np.pi, 1000)  # Interval de timp

def ex1():
    amplitudine = 1.0
    frecventa = 1.0
    faza = 0.0

    semnal_sinus = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza)

    semnal_cosinus = amplitudine * np.cos(2 * np.pi * frecventa * timp + faza - np.pi/2)

    plt.figure(figsize=(10,6))

    #Subplot pentru semnalul sinusoidal

    plt.subplot(2,1,1)
    plt.plot(timp, semnal_sinus, label="Semnal sinusoidal", color = 'blue')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Semnal sinusoidal")
    plt.xlabel('Timp (secune)')
    plt.ylabel('Amplitudine')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timp, semnal_cosinus, label="Semnal cosinusoidal", color='red')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Semnal cosinusoidal")
    plt.xlabel('Timp (secune)')
    plt.ylabel('Amplitudine')

    plt.tight_layout()
    plt.show()

def ex2():
    amplitudine = 10.0
    frecventa = 5
    faza1 = 0.2
    faza2 = 0.2
    faza3 = 0.5
    faza4 = 10
    SNR = [0.1,1,10,100]


    semnal_sinus1 = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza1)
    semnal_sinus2 = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza2)
    semnal_sinus3 = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza3)
    semnal_sinus4 = amplitudine * np.sin(2 * np.pi * frecventa * timp + faza4)

    zgomot1 = np.random.randn(len(semnal_sinus1))

    gama1 = np.sqrt(semnal_sinus1**2 / (SNR[0] * zgomot1**2))
    gama2 = np.sqrt(semnal_sinus1**2 / (SNR[1] * zgomot1**2))
    gama3 = np.sqrt(semnal_sinus1**2 / (SNR[2] * zgomot1**2))
    gama4 = np.sqrt(semnal_sinus1**2 / (SNR[3] * zgomot1**2))

    semnal_sinus1_zgomot1 =  semnal_sinus1 + gama1*zgomot1
    semnal_sinus1_zgomot2 =  semnal_sinus1 + gama2*zgomot1
    semnal_sinus1_zgomot3 =  semnal_sinus1 + gama3*zgomot1
    semnal_sinus1_zgomot4 =  semnal_sinus1 + gama4*zgomot1

    plt.subplot(2, 1, 1)
    plt.plot(timp, semnal_sinus1, label="Semnal sinusoidal", color='blue')
    plt.plot(timp, semnal_sinus2, label="Semnal sinusoidal", color='red')
    plt.plot(timp, semnal_sinus3, label="Semnal sinusoidal", color='yellow')
    plt.plot(timp, semnal_sinus4, label="Semnal sinusoidal", color='green')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Semnal sinusoidal")
    plt.xlabel('Timp (secune)')
    plt.ylabel('Amplitudine')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timp, semnal_sinus1_zgomot1, label="Semnal sinusoidal cu zgomot", color='blue')
    plt.plot(timp, semnal_sinus1_zgomot2, label="Semnal sinusoidal cu zgomot", color='red')
    plt.plot(timp, semnal_sinus1_zgomot3, label="Semnal sinusoidal cu zgomot", color='yellow')
    plt.plot(timp, semnal_sinus1_zgomot4, label="Semnal sinusoidal cu zgomot", color='green')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Semnal sinusoidal")
    plt.xlabel('Timp (secune)')
    plt.ylabel('Amplitudine')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ex1()
    ex2()

