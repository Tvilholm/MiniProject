import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import struct
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from sklearn.utils import shuffle
import sys

def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-dataset_dir':
        dataset_dir = str(args[1])	

    ## Use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    ## Load MNIST dataset
    print("Loading dataset")
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    dims = (64,64)  # dimensions of images to train/test with

    for j in range(2):  # train and test	
        for i in range(3):  # classes 0 to 2
            read_folder = dataset_dir + ('/MNIST_JPG_training/' if j==0 else '/MNIST_JPG_testing/') + str(i) + '/'
            for filename in os.listdir(read_folder):
                img = cv2.imread(os.path.join(read_folder, filename), 0)  # grayscale
                if img is not None:
                    img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
                    img = img / 255.0  # normalize
                    if j == 0:
                        train_images.append(img)
                        train_labels.append(i)
                    else:
                        test_images.append(img)
                        test_labels.append(i)

    # Convert to numpy arrays
    train_images = np.asarray(train_images, dtype='float32')
    test_images = np.asarray(test_images, dtype='float32')
    train_labels = np.asarray(train_labels, dtype='uint8')
    test_labels = np.asarray(test_labels, dtype='uint8')

    # Shuffle dataset
    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    # Define network
    model = Sequential([
        Flatten(input_shape=dims),
        Dense(32, activation='relu', use_bias=True),
        Dense(16, activation='relu', use_bias=True),
        Dense(3, activation='softmax', use_bias=True),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train network
    model.fit(train_images, train_labels, epochs=75, batch_size=64, validation_split=0.1)
    model.summary()

    # Evaluate
    start_t = time.time()
    results = model.evaluate(test_images, test_labels, verbose=0)
    totalt_t = time.time() - start_t
    print("Inference time for ", len(test_images), " test images:", totalt_t, "seconds")
    print("Test loss, test accuracy:", results)

    # Export weights to txt files (FPGA friendly)
    for w in range(1, len(model.layers)):
        weight_filename = f"layer_{w}_weights.txt"
        weights = model.layers[w].get_weights()[0]
        with open(weight_filename, 'w') as f:
            f.write('{')
            for i in range(weights.shape[0]):
                f.write('{')
                for j in range(weights.shape[1]):
                    f.write(str(weights[i][j]))
                    if j != weights.shape[1]-1:
                        f.write(', ')
                f.write('}')
                if i != weights.shape[0]-1:
                    f.write(', \n')
            f.write('}')

    # Test prediction
    x = np.expand_dims(test_images[0], axis=0)
    print("Test image[0] label:", test_labels[0])
    print("NN Prediction:", np.argmax(model.predict(x)))

    print("Finished")

if __name__ == "__main__":
    main()
