import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = pd.read_csv(r'Train.csv', delimiter=',')
x['Datetime'] = pd.to_datetime(x['Datetime'], format='%d-%m-%Y %H:%M')
datetime_values = x['Datetime'].values
count_values = x['Count'].values

def ex1_grafic():
    print(datetime_values[0])
    print(datetime_values[-1])
    print(datetime_values[:30])
    print(count_values[:30])
    plt.figure(figsize=(12,6))
    plt.plot(datetime_values, count_values,'-o', color='green')
    plt.title("Grafic - volum masini")
    plt.xlabel('Numar masini')
    plt.ylabel('Esantioane')
    plt.show()

# ex1 a) frecventa de esantionare este de aprox o ora (diferenta_timp = np.diff(datetime_values.astype(int)))
# ex1 b) interval de timp: [2012-08-25T00:00:00.000000000, 2014-09-25T23:00:00.000000000]

def ex1_c():

    # Calculați diferența de timp între eșantioane
    diferenta_timp = np.diff(datetime_values.astype(int))
    print("Diferenta de timp: ",diferenta_timp)

    # Calculați frecvența de eșantionare
    frecventa_esantionare = 1 / (np.mean(diferenta_timp))
    print("Frecventa de esantionare:",frecventa_esantionare)
    # Aplicați transformata Fourier
    frecvente = np.fft.fftfreq(len(count_values), d=1 / frecventa_esantionare)
    fft_valori = np.fft.fft(count_values)

    # Găsiți indicele valorii maxime din spectrul de amplitudine FFT
    max_frecventa_index = np.argmax(np.abs(fft_valori))

    # Găsiți frecvența corespunzătoare indicei maxim
    max_frecventa = np.abs(frecvente[max_frecventa_index])

    print(f"Frecvența maximă în semnal este aproximativ {max_frecventa:.2f} Hz.")

    # Vizualizați spectrul de frecvență
    plt.figure(figsize=(12, 6))
    plt.plot(frecvente, np.abs(fft_valori))
    plt.title('Spectrul de frecvență al semnalului')
    plt.xlabel('Frecvență (Hz)')
    plt.ylabel('Amplitudine')
    plt.show()

def ex1_d():
    # Calculați diferența de timp între eșantioane
    diferenta_timp = np.diff(datetime_values).mean().astype(int)

    # Calculați frecvența de eșantionare
    frecventa_esantionare = 1 / diferenta_timp

    # Aplicați transformata Fourier
    frecvente = np.fft.fftfreq(len(count_values), d=1 / frecventa_esantionare)
    fft_valori = np.fft.fft(count_values)

    # Vizualizați modulul transformatei Fourier
    plt.figure(figsize=(12, 6))
    plt.plot(frecvente, np.abs(fft_valori))
    plt.title('Modulul Transformatei Fourier al semnalului')
    plt.xlabel('Frecvență (Hz)')
    plt.ylabel('Amplitudine')
    plt.show()

def ex1_g():
    data_referinta = x['Datetime'][1001]
    start_date = x['Datetime'][1001 + 6 - data_referinta.weekday() + 1]
    end_date = start_date + pd.DateOffset(days=30)

    selected_data = x[(x['Datetime'] >= start_date) & (x['Datetime'] <= end_date)]
    selected_data_date_values = selected_data['Datetime'].values
    selected_data_count_values = selected_data['Count'].values

    # Plotează graficul pentru o lună de trafic
    plt.figure(figsize=(12, 6))
    plt.plot(selected_data_date_values, selected_data_count_values , label='Traffic Count')
    plt.title('Traficul pentru o lună începând de la un esantion de start > 1000 într-o zi de luni')
    plt.xlabel('Datetime')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #ex1_grafic()
    #ex1_c()
    # ex1_d()
    ex1_g()