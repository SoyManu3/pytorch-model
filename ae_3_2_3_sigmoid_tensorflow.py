import tensorflow as tf
import numpy as np
import os

# -----------------------------
# Datos
# -----------------------------
x = np.array([[1, 0, 1]], dtype=np.float32)

# -----------------------------
# Modelo Autoencoder 3-2-3
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(3,)),  # Encoder
    tf.keras.layers.Dense(3, activation='sigmoid')                      # Decoder
])

# -----------------------------
# Compilación
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss='mse'
)

# -----------------------------
# Entrenamiento
# -----------------------------
model.fit(x, x, epochs=200, verbose=1)

# -----------------------------
# Guardado del modelo (SavedModel)
# Compatible con TensorFlow Serving
# -----------------------------
export_path = "reconocimiento-mejorado/1"

os.makedirs(export_path, exist_ok=True)

# Exporta el modelo en formato SavedModel
model.export(export_path)

print("\nModelo guardado en:", export_path)

# -----------------------------
# Reconstrucción
# -----------------------------
x_hat = model.predict(x)

print("\nEntrada original:", x)
print("Reconstrucción:", x_hat)