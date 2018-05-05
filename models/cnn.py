from PIL import Image
import random
from glob import glob
import ntpath
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense


class cnn:

    def __init__(self):
        self.outPath = 'output/cnn/'

    def get_out_path(self):
        return self.outPath

    @staticmethod
    def strip_spaces(imgPath):
        # strip out spaces in image path names
        img_list = glob(imgPath + '/*')
        print("image path: ")
        print(img_list)
        for imagePath in img_list:
            print(imagePath)
            os.rename(imagePath, imagePath.replace(" copy", ""))

    @staticmethod
    def random_crop(imgPath, height, width):
        img = Image.open(imgPath)

        # check if the image dimensions are at least as big as the crop size
        imgWidth, imgHeight = img.size
        if (imgWidth < width) or (imgHeight < height):
            raise Exception('Image size is too small for the crop size')
        else:
            x = random.randint(0, imgWidth - width)
            y = random.randint(0, imgHeight - height)

            img = img.crop((x, y, x + width, y + height))
            return img

    @staticmethod
    def build_model(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(30, kernel_size=(5, 5), strides=(2, 2),
                         activation='relu',
                         input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))

        # return the constructed network architecture
        return model


def main():
    # cnn.strip_spaces('resources/casia2/Tp_patches')

    img_list = glob('resources/casia2/Au/*')
    cnnIns = cnn()
    outDir = cnnIns.get_out_path() + 'CASIA_V2/Au_cropped2'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    for imgPath in img_list:
        img = cnnIns.random_crop(imgPath, 128, 128)
        fName = outDir+os.sep+ntpath.basename(imgPath)
        fName = os.path.splitext(fName)[0]+'_2.jpg'

        img.save(fName, "JPEG")

    # img_list = glob('casia2'+os.sep+'Tp'+os.sep+'*')
    # cnnIns = cnn()
    # outDir = cnnIns.getOutPath() + 'Tp_cropped'
    # if not os.path.exists(outDir):
    #     os.makedirs(outDir)
    # for imgPath in img_list:
    #     img = cnnIns.random_crop(imgPath, 128, 128)
    #     fName = outDir+os.sep+ntpath.basename(imgPath)
    #     fName = os.path.splitext(fName)[0]+'.jpg'
    #     img.save(fName, "JPEG")

# main()
