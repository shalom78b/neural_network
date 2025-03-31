import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

     # Convert to absolute path automatically
    data_dir = os.path.abspath(sys.argv[1])
    
    if not os.path.exists(data_dir):
        sys.exit(f"Error: Path does not exist: {data_dir}")
    

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        # Skip if category directory doesn't exist
        if not os.path.isdir(category_dir):
            continue
            
        # Process each image in the category directory
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            
            # Read and resize the image
            img = cv2.imread(filepath)
            if img is not None:  # Only process if image was successfully read
                resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resized_img)
                labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layers
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Hidden layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

def main():#+
    """#+
    The main function orchestrates the entire traffic sign classification process.#+
#+
    It performs the following tasks:#+
    1. Validates the command-line arguments.#+
    2. Loads image arrays and labels for all image files from the specified data directory.#+
    3. Splits the data into training and testing sets.#+
    4. Gets a compiled neural network model.#+
    5. Fits the model on the training data.#+
    6. Evaluates the neural network performance on the testing data.#+
    7. Optionally saves the trained model to a specified file.#+
#+
    Command-line arguments:#+
    sys.argv[1] (str): The path to the directory containing the image data.#+
    sys.argv[2] (str, optional): The path to save the trained model. If not provided, the model is not saved.#+
#+
    Returns:#+
    None#+
    """#+
#+
    # Check command-line arguments#+
    if len(sys.argv) not in [2, 3]:#+
        sys.exit("Usage: python traffic.py data_directory [model.h5]")#+
#+
    # Get image arrays and labels for all image files#+
    images, labels = load_data(sys.argv[1])#+
#+
    # Split data into training and testing sets#+
    labels = tf.keras.utils.to_categorical(labels)#+
    x_train, x_test, y_train, y_test = train_test_split(#+
        np.array(images), np.array(labels), test_size=TEST_SIZE#+
    )#+
#+
    # Get a compiled neural network#+
    model = get_model()#+
#+
    # Fit model on training data#+
    model.fit(x_train, y_train, epochs=EPOCHS)#+
#+
    # Evaluate neural network performance#+
    model.evaluate(x_test,  y_test, verbose=2)#+
#+
    # Save model to file#+
    if len(sys.argv) == 3:#+
        filename = sys.argv[2]#+
        model.save(filename)#+
        print(f"Model saved to {filename}.")#+
