# Descripción de la Red Neuronal

La siguiente red neuronal está diseñada para realizar tareas de clasificación binaria utilizando texto como entrada. A continuación se detalla la estructura de la red:

## Arquitectura del Modelo

El modelo se construye utilizando la API secuencial de Keras:

```python
model = Sequential()
model.add(Input(shape=(50,)))
model.add(Embedding(v + 1, 300, weights=[embedding_matrix], trainable=False))
model.add(LSTM(20, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```
# **Capas del moodelo**
1. Input Layer:

* Descripción: Capa de entrada que espera secuencias de longitud 50.
* Dimensiones de entrada: (batch_size, 50)
2. Embedding Layer:

* Descripción: Capa de embedding pre-entrenada para representar palabras como vectores densos.
* Parámetros:
* Tamaño del vocabulario + 1: v + 1
* Dimensión de embedding: 300
* Pesos pre-entrenados: embedding_matrix (no entrenables)
3. LSTM Layer:

* Descripción: Capa LSTM con 20 unidades y return_sequences=True para mantener la salida como secuencias.
4. GlobalMaxPooling1D Layer:

* Descripción: Capa de pooling global para obtener características más relevantes de las secuencias de salida de LSTM.
5. Dense Layer (256 unidades):

* Descripción: Capa completamente conectada con 256 unidades y función de activación ReLU.
6. Dense Layer (1 unidad, activación sigmoid):
* Descripción: Capa de salida con 1 unidad y función de activación sigmoid para la clasificación binaria.

# **Compilación del Modelo**
Después de definir la arquitectura del modelo, se compila utilizando el optimizador SGD con momentum, pérdida de entropía cruzada binaria y métricas adicionales para evaluar el rendimiento:
```python
from keras.optimizers import SGD
from keras.metrics import Precision, Recall, AUC

model.compile(optimizer=SGD(0.1, momentum=0.09),
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(), Recall(), AUC()])
```
# Detalles de Compilación
* Optimizador: SGD con learning rate de 0.1 y momentum de 0.09.
* Función de Pérdida: Entropía cruzada binaria, adecuada para problemas de clasificación binaria.
* Métricas: Accuracy, Precision, Recall y AUC (Área bajo la curva ROC) para evaluar el rendimiento del modelo.
Este modelo está configurado para predecir entre dos clases (suicidio o depresión) basado en datos textuales.

