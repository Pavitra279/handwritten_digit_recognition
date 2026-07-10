import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN Model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
print("Training Model...")
model.fit(
    x_train,
    y_train,
    epochs=1,
    validation_data=(x_test, y_test)
)

# Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy * 100, "%")

# Save Model
model.save("digit_model.keras")

print("Model Saved Successfully!")

# Display Sample Prediction
sample = x_test[0]

prediction = model.predict(
    sample.reshape(1, 28, 28, 1)
)

predicted_digit = np.argmax(prediction)

plt.imshow(sample.reshape(28,28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.savefig('prediction.png', bbox_inches='tight')
plt.close()