from PIL import Image
import numpy as np
from glob import glob
import os

def img_to_array(imgPath, height, width):
    img = Image.open(imgPath)
    img = img.resize((height,width))
    imgArr = np.asarray(img)
    imgArr = np.reshape(imgArr,height * width * 3)
    return imgArr

def img_to_array_dir(dir, height, width):
    outArr = []
    img_list = glob(dir + os.sep + '*.jpg')
    for img in img_list:
        # print(img)
        outArr.append(img_to_array(img, height, width))
    return outArr


