# USAGE
# python test_model.py -i resources/casia2/Tp/Tp_D_CNN_M_N_ani00057_ani00055_11149.jpg -m output/models/cnn_ela_patches.model
# python test_model.py -v resources/videos/avengers_vfx.mp4 -m output/models/cnn_ela_auto.model

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


def sliding_window2(image, stepSize, windowSize):
    # slide a window across the image
    lastY = False
    for y in range(0, image.shape[0], stepSize):
        # y = y
        if y + stepSize > image.shape[0]:
            break
        lastX = False
        for x in range(0, image.shape[1], stepSize):
            # yield the current window

            # x = x
            if x + stepSize > image.shape[1]:
                break

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

    # print("[INFO] loading network...")
    # model = load_model(args["model"])

    category = "authentic"

    clone = image.copy()
    clone2 = cv2.imread(filename+"_diff", -1)
    (winW, winH) = (128, 128)
    for (x, y, window) in sliding_window(elaImg, stepSize=int(args["stepsize"]), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        output = window.copy()

        window = window.astype("float") / 255.0
        window = img_to_array_keras(window)
        window = np.expand_dims(window, axis=0)
        (authentic, tampered) = model.predict(window)[0]

        # build the label
        label = "Tampered" if tampered > authentic else "Authentic"
        if label == "Tampered":
            category = "Tampered"
            for xval in range(x, x+winH):
                for yval in range(y, y+winW):
                    channels = clone2[yval][xval]
                    if channels[0] > 3 or channels[1] > 3 or channels[2] > 3:
                        clone[yval][xval][2] = 255
        proba = tampered if tampered > authentic else authentic
        # label = "{}: {:.2f}%".format(label, proba * 100)
        proba = "{:.2f}%".format(proba * 100)

        # draw the label on the image

        # print("label: " + label)

        color = (0, 0, 255) if tampered > authentic else (0, 255, 0)

        cv2.rectangle(clone, (x, y), (x + winW, y + winH), color, 1)

        cv2.putText(clone, proba, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)

    cv2.imshow("Window", clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit
    # time.sleep(.125)

    os.remove(filename+"_diff")

    return clone


def perform_test_video(video, model):
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    framepath = 'tmp/video_frames/frame.jpg'
    outpath = 'output/videos/test.mp4'
    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite(framepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        outframe = perform_test(framepath, model)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        out.write(outframe)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=False,
                help="path to input image")
ap.add_argument("-v", "--video", required=False,
                help="path to input video")
ap.add_argument("-s", "--stepsize", default=128,
                help="step size for the sliding window (pixels)")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])

if args['video']:
    perform_test_video(args["video"], model)
elif args['image']:
    perform_test(args['image'], model)
else:
    print("Please provide either an image or a video")