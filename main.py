import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D, Dropout, UpSampling2D, Concatenate

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


from sklearn.model_selection import train_test_split

# Generowanie unikalnych współrzędnych
def generate_unique_coordinates(num_samples):
    population = np.arange(64)
    coordinates = []

    while len(coordinates) < num_samples:
        # Losowanie czterech współrzędnych
        coords = np.random.choice(population, size=4, replace=False)
        coords = coords.tolist()

        # Sprawdzenie, czy ta konfiguracja współrzędnych już istnieje
        if coords not in coordinates:
            coordinates.append(coords)

    return np.array(coordinates)

# Generowanie danych
coordinates = generate_unique_coordinates(10000)

# Podział na dane treningowe i testowe w proporcji 2/8
train_coordinates, test_coordinates = train_test_split(coordinates, test_size=0.8, random_state=42)


print("Dane treningowe:")
print(train_coordinates)

print("\nDane walidacyjne:")
print(test_coordinates)


# Przygotowanie danych treningowych
x_train = np.zeros((train_coordinates.shape[0], 64, 64))
y_train = np.zeros((train_coordinates.shape[0], 4))
for i, coords in enumerate(train_coordinates):
    x_train[i, coords[0], coords[1]] = 1
    x_train[i, coords[2], coords[3]] = 1
    y_train[i] = coords

# Przygotowanie danych testowych
x_test = np.zeros((test_coordinates.shape[0], 64, 64))
y_test = np.zeros((test_coordinates.shape[0], 4))
for i, coords in enumerate(test_coordinates):
    x_test[i, coords[0], coords[1]] = 1
    x_test[i, coords[2], coords[3]] = 1
    y_test[i] = coords


# Tworzenie sztucznych danych uczących
# def generate_data(num_samples):
#     X_train = np.zeros((num_samples, 64, 64, 1))
#     y_train = np.zeros((num_samples, 4))  # cztery współrzędne: x_początek, y_początek, x_koniec, y_koniec
#
#     for i in range(num_samples):
#         # Generowanie losowej linii na obrazie
#         x_start = np.random.randint(0, 64)
#         y_start = np.random.randint(0, 64)
#         x_end = np.random.randint(0, 64)
#         y_end = np.random.randint(0, 64)
#
#         # Tworzenie obrazu z linią
#         img = Image.new('L', (64, 64), color=255)
#         img_draw = ImageDraw.Draw(img)
#         img_draw.line([(x_start, y_start), (x_end, y_end)], fill=0, width=1)
#
#         # Konwersja obrazu do tablicy numpy
#         img_array = np.array(img)
#         img_array = img_array.reshape((64, 64, 1))
#
#         X_train[i] = img_array
#         y_train[i] = [x_start, y_start, x_end, y_end]
#
#     return X_train, y_train

# Generowanie danych uczących
# X_train, y_train = generate_data(10000) #generate_ training_data
# X_val, y_val = generate_data(2000) #generate_validation_data

# Tworzenie modelu sieci neuronowej
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(4))  # 4 wyjścia: x_początek, y_początek, x_koniec, y_koniec
# Tworzenie modelu

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),  # Spłaszczanie danych do 1D
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(4)  # Wyjściowa warstwa z 4 neuronami (współrzędnymi)
])
# przy tym modelu dokładność danych walidacyjnych również rośnie
# loss = MSE spada --> model się uczy

# Kompilowanie i uczenie modelu
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# Trenowanie modelu
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Testowanie modelu na przykładowym obrazie
sample_image = Image.new('L', (64, 64), color=255)
sample_image_draw = ImageDraw.Draw(sample_image)
sample_image_draw.line([(10, 20), (40, 50)], fill=0, width=1)  # symulacja linii na obrazie

# Konwersja obrazu do tablicy numpy
sample_image_array = np.array(sample_image)
sample_image_array = sample_image_array.reshape((1, 64, 64, 1))
predictions = model.predict(sample_image_array)

# Wyświetlanie wyników
print("Przewidziane współrzędne:")
print("x_początek:", predictions[0][0])
print("y_początek:", predictions[0][1])
print("x_koniec:", predictions[0][2])
print("y_koniec:", predictions[0][3])

# Wykresy dokładności i straty
plt.figure(figsize=(12, 4))

# Dokładność treningu i walidacji - wykres Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Dokładność treningu i walidacji - wykres mean_squared_error
# plt.plot(history.history['loss'], label='Train mean_squared_error')
# plt.plot(history.history['val_loss'], label='Validation mean_squared_error')
# plt.xlabel('Epochs')
# plt.ylabel('mean_squared_error')
# plt.legend()
# plt.show()
