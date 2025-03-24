import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0  # Normalize training data
x_test = x_test / 255.0    # Normalize testing data

# Define the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Make a prediction on a random test sample
sample_index = np.random.choice(len(x_test))
sample_image = x_test[sample_index]
sample_label = y_test[sample_index]

sample_input = sample_image.reshape((1, 28, 28))  # Reshape for the model input
prediction = model.predict(sample_input)

predicted_label = np.argmax(prediction)
print(predicted_label)

# Class labels for interpretation
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Get the predicted and true object names
predicted_object = class_labels[predicted_label]
true_object = class_labels[sample_label]

# Display the results
plt.imshow(sample_image, cmap='gray')
plt.title(f'True Label: {sample_label} ({true_object}), Predicted Label: {predicted_label} ({predicted_object})')
plt.axis('off')
plt.show()

# Print the predicted object and label
print(f"The predicted object is {predicted_object} and label is {predicted_label}")