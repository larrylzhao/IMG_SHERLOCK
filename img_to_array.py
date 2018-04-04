from PIL import Image
import numpy as np
from glob import glob
import os
from keras.preprocessing.image import img_to_array as keras_img_to_array
import ntpath

def img_to_array(img, height, width):
    imgArr = np.asarray(img)
    imgArr = np.reshape(imgArr, height * width * 3)
    return imgArr

def img_to_array_from_path(imgPath, height, width):
    img = Image.open(imgPath)
    img = img.resize((height, width))
    imgArr = np.asarray(img)
    imgArr = np.reshape(imgArr, height * width * 3)
    return imgArr

def img_to_array_dir(dir, height, width):
    outArr = []
    img_list = glob(dir + os.sep + '*')
    for img in img_list:
        # print(img)
        outArr.append(img_to_array_from_path(img, height, width))
    return outArr

def keras_img_to_array_dir(dir, height, width):
    outArr = []
    img_list = glob(dir + os.sep + '*')
    for imgPath in img_list:

        img = Image.open(imgPath)
        img = img.resize((height, width))
        img = keras_img_to_array(img)
        outArr.append(img)
        # print(img)
    return outArr

def image_resize(dir, height, width):
    img_list = glob(dir + os.sep + '*')
    for imgPath in img_list:

        img = Image.open(imgPath)
        img = img.resize((height, width))
        filename = "resources/casia2/Tp_128_128"+os.sep+ntpath.basename(imgPath)
        img.save(os.path.splitext(filename)[0]+'.jpg', "JPEG")


# image_resize('resources/casia2/Tp', 128, 128)
