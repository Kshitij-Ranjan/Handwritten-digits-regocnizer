# Basic libraries
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow & Keras for ML
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

np.set_printoptions(linewidth=300, threshold=4000)

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check the shapes
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

#Look at the 122nd image in the training set
print("Label:", y_train[122])  # The correct digit
print("Image matrix:\n", x_train[122])

#shows the dataset
plt.figure(figsize=(10,4))  # Make the figure bigger for clarity
for i in range(20):                     # Loop through first 10 images
    plt.subplot(3,10,i+1)         # Arrange images in 2 rows, 5 columns
    plt.imshow(x_train[i], cmap='gray') # Show image in grayscale
    plt.title(y_train[i])               # Display the correct digit as title
    plt.axis('off')                     # Remove axes for a cleaner look
plt.show()  # Render the images


# Normalize pixel values to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot vectors
y_train = to_categorical(y_train) # 3 becomes [0,0,0,1,0,0,0,0,0,0], 1 representing probability of being 3 and the index of 1 is 3 as 0 1 2 3
y_test = to_categorical(y_test)

model = Sequential([
    Input(shape=(28,28)),           # Proper way to define input shape
    Flatten(),                       # Flatten input to 1D
    Dense(128, activation='relu'),   # Hidden layer #128
    Dense(10, activation='softmax')  # Output layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,      # Training data (images + labels)
    epochs=10,             # Number of times to go through the entire dataset
    batch_size=32,         # Number of samples processed before updating weights
    validation_split=0.2   # 20% of training data used for validation
)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
