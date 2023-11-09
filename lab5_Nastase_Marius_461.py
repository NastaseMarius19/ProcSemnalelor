
import matplotlib.pyplot as plt
import pandas as pd

def ex1():
    x = pd.read_csv(r'Train.csv', delimiter=',')
    x['Datetime'] = pd.to_datetime(x['Datetime'], format='%d-%m-%Y %H:%M')

    plt.figure(figsize=(12,6))
    plt.plot(x['Datetime'], x['Count'], linefmt = '-', markerfmt='o', basefmt='k')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title("Grafic - volum masini")
    plt.xlabel('Datetime')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ex1()