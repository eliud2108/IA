import numpy as np
from keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import cv2
import numpy as np
from keras import models
import serial
import time

# Cargar el conjunto de datos
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Preprocesamiento de datos
x_train = train_data.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = test_data.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Crear el modelo CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Evaluar el modelo
model.evaluate(x_test, y_test)




# Cargar el modelo entrenado
#model = models.load_model('modelo_entrenado.h5')

# Inicializar la cámara USB
cap = cv2.VideoCapture(1)

# Inicializar la comunicación serial con Arduino
arduino = serial.Serial('COM3', 9600)
#arduino.write(b's')  # Enviar un comando para iniciar

while True:
    # Capturar imagen desde la cámara
    ret, frame = cap.read()
    # Verificar si se pudo capturar el fotograma
    if not ret:
        print("No se pudo capturar el fotograma")
        break
    # Preprocesar la imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.astype('float32') / 255.
    img = np.expand_dims(img, axis=0)
    
    # Realizar la predicción
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    
    #cv2.putText(frame, 'Prediccion: {}'.format(predicted_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.imshow('Captura de imagen y prediccion', frame)
    
    # Verificar si el número predicho es par o impar
    if predicted_label % 2 == 0:
        par_impar = "par"
        arduino.write(b'p')  # Enviar comando a Arduino para indicar que el número es par
        time.sleep(7)  # Esperar 7 segundos
        arduino.write(b'c')  # Enviar comando a Arduino para indicar que el servo debe moverse a la posición central
        time.sleep(5) 
    else:
        par_impar = "impar"
        arduino.write(b'i')  # Enviar comando a Arduino para indicar que el número es impar
        time.sleep(7)  # Esperar 7 segundos
        arduino.write(b'c')  # Enviar comando a Arduino para indicar que el servo debe moverse a la posición central
        time.sleep(5) 
    
    # Mostrar la imagen y el número predicho en la misma ventana
    #cv2.putText(frame, 'Predicción: {} ({})'.format(predicted_label, par_impar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.imshow('Captura de imagen y predicción', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or not cap.isOpened():
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
