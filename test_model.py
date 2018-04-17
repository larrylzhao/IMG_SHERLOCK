# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

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


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    lastY = False
    for y in range(0, image.shape[0], stepSize):
        # y = y
        if y + windowSize[1] + stepSize > image.shape[0]:
            y = image.shape[0] - windowSize[1]
            lastY = True
        lastX = False
        for x in range(0, image.shape[1], stepSize):
            # yield the current window

            # x = x
            if x + windowSize[0] + stepSize > image.shape[1]:
                x = image.shape[1] - windowSize[0]
                lastX = True

            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            if lastX:
                break
        if lastY:
            break


def perform_test():
    # load the image
    image = cv2.imread(args["image"])
    if image is None:
        raise Exception("image cannot be read or does not exist")
        exit()

    orig = Image.open(args["image"])

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

    print("[INFO] loading network...")
    model = load_model(args["model"])

    category = "authentic"

    clone = image.copy()
    (winW, winH) = (128, 128)
    for (x, y, window) in sliding_window(elaImg, stepSize=int(args["stepsize"]), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        output = window.copy()

        # since we do not have a classifier, we'll just draw the window



        window = window.astype("float") / 255.0
        window = img_to_array_keras(window)
        window = np.expand_dims(window, axis=0)
        (authentic, tampered) = model.predict(window)[0]

        # build the label
        label = "Tampered" if tampered > authentic else "Authentic"
        if label == "Tampered":
            category = tampered
        proba = tampered if tampered > authentic else authentic
        label = "{}: {:.2f}%".format(label, proba * 100)
        proba = label = "{:.2f}%".format(proba * 100)

        # draw the label on the image

        print("label: " + label)

        color = (0, 0, 255) if tampered > authentic else (0, 255, 0)


        cv2.rectangle(clone, (x, y), (x + winW, y + winH), color, 2)
        # cv2.putText(clone, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (0, 255, 0), 2)

        cv2.putText(clone, proba, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(0)
        time.sleep(.125)

    os.remove(filename+"_diff")

    return category

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-s", "--stepsize", default=128,
                help="step size for the sliding window (pixels)")
args = vars(ap.parse_args())

perform_test()