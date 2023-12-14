import numpy as np
import matplotlib.pyplot as plt

# Dimensiunea seriei de timp
N = 1000

# Generăm componente aleatorii pentru trend, sezon și variabilitate mică
t = np.arange(N)
trend = 0.02 * t**2  # Componenta trend, ecuație de grad 2
seasonal = 10 * np.sin(0.02 * np.pi * t) + 5 * np.sin(0.04 * np.pi * t)  # Două componente sezoniere
noise = np.random.normal(0, 5, N)  # Variatiile mici, zgomot alb gaussian

# Sumăm cele trei componente pentru a obține seria de timp
time_series = trend + seasonal + noise

def calcS_t(alpha, t, time_series):
    suma = 0
    for i in range(t):
        suma = 0
        for j in range(i):
            suma = suma +   (1-alpha)**(i+1-j) * time_series[j]
    suma = alpha*suma
    suma = suma + ((1-alpha)**t)*time_series[0]
    return suma
def ex2():
    s = np.zeros(len(time_series))
    for i in range(len(time_series)):
        s[i] = calcS_t(0.95,i,time_series)

    plt.subplot(4, 1, 1)
    plt.plot(t, time_series, label='Seria de timp')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, s, label='Seria de timp')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ex2()

