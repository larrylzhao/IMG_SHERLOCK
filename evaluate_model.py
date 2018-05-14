# USAGE
# python evaluate_model.py -a resources/casia2/Au -t resources/casia2/Tp -m output/models/cnn_ela.model

# import the necessary packages
from keras.preprocessing.image import img_to_array as img_to_array_keras
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import os
from PIL import Image, ImageChops
from io import BytesIO
import random
from imutils import paths


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    lastY = False
    for y in range(0, image.shape[0], stepSize):
        # y = y
        if y + stepSize > image.shape[0]:
            y = image.shape[0] - windowSize[1]
            lastY = True
        lastX = False
        for x in range(0, image.shape[1], stepSize):
            # yield the current window

            # x = x
            if x + stepSize > image.shape[1]:
                x = image.shape[1] - windowSize[0]
                lastX = True

            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            if lastX:
                break
        if lastY:
            break


def perform_test(testimage, model):
    # load the image
    image = cv2.imread(testimage)
    if image is None:
        raise Exception("image cannot be read or does not exist")
        exit()

    orig = Image.open(testimage)

    # resize image if too large
    maxDimension = 500
    w, h = orig.size
    # if w > maxDimension or h > maxDimension:
    # ratio = min(maxDimension/w, maxDimension/h)
    # dim = (int(w*ratio), int(h*ratio))
    # image = cv2.resize(image, dim)
    # orig = orig.resize(dim)

    # perform ELA on the image
    outDir = "tmp/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    elaImg = orig
    filename = outDir + uuid.uuid4().hex[:6].upper()
    byte_io = BytesIO()
    elaImg.save(byte_io, "JPEG", quality=90)
    elaImg = Image.open(byte_io)
    elaImg = ImageChops.difference(orig, elaImg)
    elaImg.save(filename+"_diff", "JPEG")
    elaImg = cv2.imread(filename+"_diff")

    (winW, winH) = (128, 128)
    for (x, y, window) in sliding_window(elaImg, stepSize=int(args["stepsize"]), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        window = window.astype("float") / 255.0
        window = img_to_array_keras(window)
        window = np.expand_dims(window, axis=0)
        (authentic, tampered) = model.predict(window)[0]

        # build the label
        label = "Tampered" if tampered > authentic else "Authentic"
        if (label == "Tampered"):
            # cv2.imshow("Window", window)
            # cv2.waitKey(0)
            return "Tampered"



    os.remove(filename+"_diff")

    return "Authentic"


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--auset", required=True,
                help="path to input authentic dataset")
ap.add_argument("-t", "--tpset", required=True,
                help="path to input tampered dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-s", "--stepsize", default=128,
                help="step size for the sliding window (pixels)")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])

# grab the image paths and randomly shuffle them
auImagePaths = sorted(list(paths.list_images(args["auset"])))
tpImagePaths = sorted(list(paths.list_images(args["tpset"])))

random.seed(42)
random.shuffle(auImagePaths)
random.shuffle(tpImagePaths)

# limit size of dataset
auImagePaths = auImagePaths[:1000]

# balance the datasets
if (len(auImagePaths) < len(tpImagePaths)):
    tpImagePaths = tpImagePaths[:len(auImagePaths)]
else:
    auImagePaths = auImagePaths[:len(tpImagePaths)]

print("[INFO] evaluating authentic set...")
auCnt = 0
tpCnt = 0
for imagePath in auImagePaths:
    classification = perform_test(imagePath, model)
    if classification == "Tampered":
        tpCnt += 1
    else:
        auCnt += 1

auAcc = float(auCnt/len(auImagePaths) * 100)
print("AU: " + str(auCnt))
print("TP: " + str(tpCnt))
print("AC: " + str(auAcc))

print("[INFO] evaluating tampered set...")
auCnt = 0
tpCnt = 0
for imagePath in tpImagePaths:
    classification = perform_test(imagePath, model)
    if classification == "Tampered":
        tpCnt += 1
    else:
        auCnt += 1

tpAcc = float(tpCnt/len(tpImagePaths) * 100)
print("AU: " + str(auCnt))
print("TP: " + str(tpCnt))
print("AC: " + str(tpAcc))

acc = float((auAcc + tpAcc)/2)
print("total accuracy: " + str(acc))
