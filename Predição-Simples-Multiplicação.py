import numpy as np
import tensorflow as tf

# Dados de treinamento
X_train = np.array([[5,7], [8,9], [2,3], [2,1], [3,3], [7, 6], [5,4], [2,2]], dtype=float)
y_train = np.array([[35], [72], [6], [2], [9], [42], [20], [4]], dtype=float)

# Criando o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)  # Saída única
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Treinando o modelo
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Testando o modelo
test_data = np.array([[5, 6], [2, 2], [7, 8]], dtype=float)
predictions = model.predict(test_data)

# Exibindo os resultados
for i in range(len(test_data)):
    print("Entrada:", test_data[i], " Saída Prevista:", predictions[i])
