from sklearn.model_selection import train_test_split
import cv2
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import datetime

#function to load images to an array
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb
        img = cv2.resize(img,(64,64)) #resize the image to 64x64x3
        if img is not None:
            images.append(img)
    return images

#function to load labels to an array
def load_labels(file):
    labels = []
    file1 = open(file, 'r')
    Lines = file1.readlines()
    for line in Lines:
        labels.append(int(line[0])) #add the first char which is the lab to the labels array
    return labels

#function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                    title='Confusion matrix',
                    cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def create_model(input_shape):
    # network found in the paper
    convSize = 3

    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        keras.layers.Conv2D(filters=64, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        keras.layers.Conv2D(filters=128, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        keras.layers.Conv2D(filters=256, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.Conv2D(filters=256, kernel_size=convSize, activation="relu", input_shape=input_shape,
                            padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    images = load_images("./genki/files")  #load all images into an array
    labels = load_labels("./genki/labels.txt")  #load all labels into an array
    (train_x, test_x, train_y, test_y) = train_test_split(images, labels, test_size=0.20, random_state=1)  #Split the dataset into fixed training and testing sets at random (80%/20%)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Normalize data
    train_x = train_x.astype(np.float32) / 255.
    test_x = test_x.astype(np.float32) / 255.

    #take 100 images for validation
    val_x = train_x[-100:]
    val_y = train_y[-100:]
    train_x = train_x[:-100]
    train_y = train_y[:-100]


    model = create_model(input_shape=train_x.shape[1:])

    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer= opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(val_x, val_y), callbacks=[tensorboard_callback]) #train the model for 50 epochs
    model.save("./models/model") #save the model named merakan_model
    score = model.evaluate(test_x, test_y)

    print('Test loss:', score[0]) #printing the test results
    print('Test accuracy:', score[1])

    #plotting the training loss and accuracy
    plt.figure()
    plt.plot(history.history["loss"], label="training loss", color = "r")
    plt.plot(history.history["accuracy"], label="training accuracy", color = "b")
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy or Loss")
    plt.legend()
    plt.show()

    #plotting confusion matrix
    predictions = []
    modelrestored = keras.models.load_model("./models/model")
    for i in test_x:
        i = i[np.newaxis, :]
        predictions.append(int(modelrestored.predict(i).round()))

    cm = confusion_matrix(y_true=test_y, y_pred=predictions)

    class_names = ['unsmiling', 'smiling']

    plot_confusion_matrix(cm=cm, classes=class_names, title='Confusion Matrix')

if __name__ == "__main__":
    main()