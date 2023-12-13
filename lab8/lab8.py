import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Dimensiunea seriei de timp
N = 1000

# Generăm componente aleatorii pentru trend, sezon și variabilitate mică
t = np.arange(N)
trend = 0.02 * t**2  # Componenta trend, ecuație de grad 2
seasonal = 10 * np.sin(0.02 * np.pi * t) + 5 * np.sin(0.04 * np.pi * t)  # Două componente sezoniere
noise = np.random.normal(0, 5, N)  # Variatiile mici, zgomot alb gaussian

# Sumăm cele trei componente pentru a obține seria de timp
time_series = trend + seasonal + noise
def ex1():

    # Desenăm seriile de timp și componente separat
    plt.figure(figsize=(12, 6))

    plt.subplot(4, 1, 1)
    plt.plot(t, time_series, label='Seria de timp')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, trend, label='Trend')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, seasonal, label='Sezon')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, noise, label='Variatiile mici')
    plt.legend()

    plt.tight_layout()
    plt.show()

def ex2():
    # Calculăm autocorelația
    autocorrelation = np.correlate(time_series, time_series, mode='full')

    # Normalizează autocorelația pentru a obține coeficienții de autocorelație
    autocorrelation /= np.max(autocorrelation)

    # Desenează vectorul de autocorelație
    plt.figure(figsize=(12, 4))
    plt.plot(autocorrelation)
    plt.title('Vectorul de Autocorelație')
    plt.xlabel('Întârzieri')
    plt.ylabel('Autocorelație normalizată')
    plt.show()

    '''
    Autocorelația măsoară gradul de corelație între o serie de timp și versiunea sa întârziată. Atunci când calculăm 
    autocorelația pentru o serie de timp la diferite întârzieri, obținem un vector de autocorelație. Acest vector arată 
    cât de corelate sunt valorile seriei de timp la diferite momente în timp, precum și gradul de repetare al modelelor.
    '''

def ex3():
    # Calculăm modelul AR cu o anumită ordine p
    p = 10
    model = ARIMA(time_series, order=(p, 0, 0))
    result = model.fit()

    # Obținem predictiile modelului AR
    predictions = result.predict(start=p, end=N - 1)

    # Afișăm seria de timp originală și predictiile
    plt.figure(figsize=(12, 6))
    plt.plot(t, time_series, label='Seria de timp originală')
    plt.plot(t[p:], predictions, label=f'Predictii AR(p={p})', color='red', linestyle='dashed')
    plt.legend()
    plt.title('Seria de timp originală și predictiile AR')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.show()

if __name__ == "__main__":
    # ex1()
    # ex2()
    ex3()
