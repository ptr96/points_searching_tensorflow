from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split

# ## Test ustawiania współrzędnych
# coords = np.array([10, 4, 10, 2])
# corner_1, corner_2 = tuple(coords[:2]), tuple(coords[2:])
# coords = min(corner_1, corner_2) + max(corner_1, corner_2)
# print(coords)

# Generowanie unikalnych współrzędnych
def generate_unique_coordinates(num_samples):
    np.random.seed(100)
    population = np.arange(64)
    coordinates = []

    #losujemy do póki nie będzie num_samples unikalnych koordynat
    while len(coordinates) < num_samples:
        # Losowanie czterech współrzędnych
        coords = np.random.choice(population, size=4)
        # ustawienie współrzędnych parami: najpierw x_początku < x_końca, jeśli ok to ustawione if  x_początku == x_końca
        # to y_początku < y_końca
        # wspólrzedne musimy ustawić, w przeciwnym wypadku model sie nie uczy.
        corner_1, corner_2 = tuple(coords[:2]), tuple(coords[2:])
        coords = min(corner_1, corner_2) + max(corner_1, corner_2)
        #zebranie wszystkich koordynat do jednej listy
        coords = list(coords)
        # Sprawdzenie, czy ta konfiguracja współrzędnych już istnieje
        if coords not in coordinates:
            coordinates.append(coords)

    return np.array(coordinates)


# Generowanie danych

coordinates = generate_unique_coordinates(50000)

# Podział na dane treningowe i testowe w proporcji 2/8
train_coordinates, test_coordinates = train_test_split(coordinates, test_size=0.2, random_state=42)

# Test --> jaką tablicę otrzymujemy
# print(train_coordinates.shape)
# print(test_coordinates.shape)

# Test --> koordynaty pierwszej generowanej losowo tablicy
# print(train_coordinates[0])

# for i in range(30):
#     print(train_coordinates[i])

#Tworzenie obrazków
def generate_img(coordinates):
    img = Image.new('L', (64, 64), color=0)
    img_draw = ImageDraw.Draw(img)
    corner_1, corner_2 = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
    img_draw.line([corner_1, corner_2], fill=1, width=1)
    img_array = np.array(img)
    return img_array

#Funkcja wyświetlania linii na układzie współrzędnych na potrzeby sprawdzenia poprawności wprowadzanych danych
def show_img(img):
  plt.imshow(img, interpolation='none')
  plt.show()

# Test --> czy generowany obraz jest własciwy
# show_img(generate_img(train_coordinates[0]))

def generate_images(coordinates_list):
  x = []
  for coordinates in coordinates_list:
    img_array = generate_img(coordinates)
    x.append(img_array)
  return np.array(x), coordinates_list

x_train, y_train = generate_images(train_coordinates)
x_test, y_test = generate_images(test_coordinates)


#sprawdzenie poprawności rozmiarów tablic danych
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

#sprawdzenie poprawności danych treningowych
# show_img(x_train[5])
# print obrazka dla danych 5
# print(y_train[5])
# print danych 5

#sprawdzenie poprawności danych validacyjnych
# show_img(x_test[0])
# print(y_test[0])


## -----------------------"""Model sieci neuronowej"""

## --------------------- MODEL W WERSJI Z UŻYCIEM SIECI KONWOLUCYJNYCH --------
## (zalecane przy problemach zwiazanych z obrazem)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
# # model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
# model.add(Flatten())
# keras.layers.Dropout(0.2),
# model.add(Dense(32, activation='relu'))
# model.add(Dense(4))  # 4 wyjścia: x_początek, y_początek, x_koniec, y_koniec


## --------------------- MODEL W WERSJI Z UŻYCIEM SIECI KONWOLUCYJNYCH  (szybszy i prostszy ) --------

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),  # Spłaszczanie danych do 1D
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4)  # Wyjściowa warstwa z 4 neuronami (współrzędnymi)
])

## Kompilopwanie modelu z odpowiednimi metrykami - metryki do podania skuteczności
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

## Trenowanie modelu
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Dokładność treningu i walidacji - wykres mean_squared_error
plt.plot(history.history['loss'], label='Train mean_squared_error')
plt.plot(history.history['val_loss'], label='Validation mean_squared_error')
plt.xlabel('Epochs')
plt.ylabel('mean_squared_error')
plt.legend()
plt.show()

# Dokładność treningu i walidacji - wykres sredni błąd bezwzględny
plt.plot(history.history['mean_absolute_error'], label='Train mean absolute error')
plt.plot(history.history['val_mean_absolute_error'], label='Validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('Absolute error')
plt.legend()
plt.show()


## Sprawdzenie wyników predykcji modelu sieci neuronowej
preds = model.predict(x_test)
preds_train = model.predict(x_train)


# print(y_test[0])
# print(preds[0])
# show_img(x_test[0])


# print(y_test[100])
# print(preds[100])
# show_img(x_test[100])

## Porównanie tabel
# for i in range(20):
#   print(y_test[i], preds[i])
# for i in range(20):
#   print(y_train[i], preds_train[i])
