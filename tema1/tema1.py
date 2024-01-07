import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from skimage import color
from scipy.datasets import face, ascent

# imagine din scipy
X = ascent()

def afisImagineX():
    plt.imshow(X, cmap=plt.cm.gray)
    plt.show()

# transformata DCT a imaginii
Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)\

def transformataDCT():
    freq_db_1 = 20 * np.log10(abs(Y1))
    freq_db_2 = 20 * np.log10(abs(Y2))
    freq_db_3 = 20 * np.log10(abs(Y3))
    freq_db_4 = 20 * np.log10(abs(Y4))

    plt.subplot(221).imshow(freq_db_1)
    plt.subplot(222).imshow(freq_db_2)
    plt.subplot(223).imshow(freq_db_3)
    plt.subplot(224).imshow(freq_db_4)
    plt.show()

def compactareImagine():
    k = 120

    Y_ziped = Y2.copy()
    Y_ziped[k:] = 0
    X_ziped = idctn(Y_ziped)

    plt.imshow(X_ziped, cmap=plt.cm.gray)
    plt.show()

def JPEG():
    Q_down = 10

    X_jpeg = X.copy()
    X_jpeg = Q_down * np.round(X_jpeg / Q_down);

    plt.subplot(121).imshow(X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
    plt.title('Down-sampled')
    plt.show()

# pentru fiecare bloc de 8*8 aplica DCT si cuantizare

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

def q_jpeg():
    # Encoding
    x = X[:8, :8]
    y = dctn(x)
    y_jpeg = Q_jpeg * np.round(y / Q_jpeg)

    # Decoding
    x_jpeg = idctn(y_jpeg)

    # Results
    y_nnz = np.count_nonzero(y)
    y_jpeg_nnz = np.count_nonzero(y_jpeg)

    plt.subplot(121).imshow(x, cmap=plt.cm.gray)
    plt.title('Original')
    plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
    plt.title('JPEG')
    plt.show()

    print('Componente în frecvență:' + str(y_nnz) +
          '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

#  Completați algoritmul JPEG incluzând toate blocurile din imagine.
def ex1():
    lungime_bloc = 8
    linii, coloane = X.shape
    for i in range(0, linii, lungime_bloc):
        for j in range(0, coloane, lungime_bloc):
            # luam un bloc
            bloc = X[i:i+lungime_bloc, j:j+lungime_bloc]

            y = dctn(bloc)

            y_jpeg = Q_jpeg * np.round(y / Q_jpeg)

            x_jpeg = idctn(y_jpeg)

            X[i:i+lungime_bloc, j:j+lungime_bloc] = x_jpeg

    plt.subplot(121).imshow(X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
    plt.title('JPEG')
    plt.show()

def ex2():
    imagine_rgb = face()
    imagine_ycbcr = color.rgb2ycbcr(imagine_rgb)
    lungime_bloc = 8
    linii, coloane, _ = imagine_rgb.shape

    for i in range(0, linii, lungime_bloc):
        for j in range(0, coloane, lungime_bloc):
            # luam un bloc de lungime data
            bloc = imagine_ycbcr[i:i+lungime_bloc,j:j+lungime_bloc,0]

            # aplicam DCT
            y = dctn(bloc)

            y_jpeg = Q_jpeg * np.round(y / Q_jpeg)

            x_jpeg = idctn(y_jpeg)

            imagine_ycbcr[i:i+lungime_bloc, j:j+lungime_bloc,0] = x_jpeg

    # transformam imaginea Y'CbCr inapoi in RGB
    imagine_jpeg_rgb = color.ycbcr2rgb(imagine_ycbcr)

    # Afișează rezultatul
    plt.subplot(121).imshow(imagine_jpeg_rgb)
    plt.title('Original')
    plt.subplot(122).imshow(x_jpeg)
    plt.title('JPEG')
    plt.show()


def compress_image_ex3(mse_threshold):
    X_ = ascent()
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]
    lungime_bloc = 8
    linii, coloane = X_.shape
    X_compressed = np.copy(X_)  # Copiem imaginea originală pentru a nu o modifica
    mse_current = float('inf')  # Inițializăm MSE cu o valoare mare

    # Ajustăm Q_jpeg pentru a îndeplini pragul MSE
    while mse_current > mse_threshold:
        for i in range(0, linii, lungime_bloc):
            for j in range(0, coloane, lungime_bloc):
                bloc = X[i:i + lungime_bloc, j:j + lungime_bloc]
                y = dctn(bloc)
                y_jpeg = Q_jpeg * np.round(y / Q_jpeg)
                x_jpeg = idctn(y_jpeg)
                X_compressed[i:i + lungime_bloc, j:j + lungime_bloc] = x_jpeg

        mse_current = np.mean((X- X_compressed) **2)

        # Dacă MSE este încă prea mare, mărim Q_jpeg
        if mse_current > mse_threshold:
            Q_jpeg += 1

    return X_compressed

def ex3():
    mse_ales = 100
    imagine_compactata = compress_image_ex3(mse_ales)

    # Afișează imaginea originală și imaginea comprimată
    plt.subplot(121).imshow(X)
    plt.title('Original Image')
    plt.subplot(122).imshow(imagine_compactata)
    plt.title(f'Compressed Image (MSE < {mse_ales})')
    plt.show()

if __name__ == '__main__':
    #afisImagineX()
    #transformataDCT()
    #compactareImagine()
    #JPEG()
    #q_jpeg()
    #ex1()
    #ex2()
    ex3()