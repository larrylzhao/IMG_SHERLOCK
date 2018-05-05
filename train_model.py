# modeled after https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array as img_to_array_keras
from keras.utils import to_categorical
from models.cnn import cnn
from models.ela import ela
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import tensorflow as tf
import os
from img_to_array import img_to_array

# try to fix bug with GPU tensorflow
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--auset", required=True,
                help="path to input authentic dataset")
ap.add_argument("-t", "--tpset", required=True,
                help="path to input tampered dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-al", "--algorithm", default="cnn",
                help="algorithm to use (cnn/ela)")
args = vars(ap.parse_args())

# initialize model parameters
SPLIT = .75
EPOCHS = 100
LR = .001
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
auImagePaths = sorted(list(paths.list_images(args["auset"])))
tpImagePaths = sorted(list(paths.list_images(args["tpset"])))

random.seed(42)
random.shuffle(auImagePaths)
random.shuffle(tpImagePaths)

# limit size of dataset
# auImagePaths = auImagePaths[:800]

# balance the datasets
if (len(auImagePaths) < len(tpImagePaths)):
    tpImagePaths = tpImagePaths[:len(auImagePaths)]
else:
    auImagePaths = auImagePaths[:len(tpImagePaths)]

# create dataset and labels
for imagePath in auImagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    if args["algorithm"] == "cnn":
        image = img_to_array_keras(image)
    elif args["algorithm"] == "ela":
        image = img_to_array(image,  128, 128)
    data.append(image)
    labels.append(0)
for imagePath in tpImagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    if args["algorithm"] == "cnn":
        image = img_to_array_keras(image)
    elif args["algorithm"] == "ela":
        image = img_to_array(image, 128, 128)
    data.append(image)
    labels.append(1)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=1 - SPLIT, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = Sequential()
if args["algorithm"] == "cnn":
    model = cnn.build_model(width=128, height=128, depth=3, classes=2)
elif args["algorithm"] == "ela":
    model = ela.build_model(dim=trainX.shape[1], classes=2)
opt = Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model
if args["algorithm"] == "cnn":
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)
elif args["algorithm"] == "ela":
    # train model without using data augmentation
    H = model.model.fit(trainX, trainY,
                        epochs=EPOCHS,
                        batch_size=BS,
                        verbose=1,
                        validation_data=(testX, testY))

# save the model to disk
print("[INFO] serializing network...")
modelParentPath = os.path.abspath(os.path.join(args["model"], os.pardir))
if not os.path.exists(modelParentPath):
    os.makedirs(modelParentPath)
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])