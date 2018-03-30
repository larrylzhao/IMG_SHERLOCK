from PIL import Image
import random
from glob import glob
import ntpath
import os


class cnn:

    def __init__(self):
        self.outPath = 'cnn'+os.sep+'CASIA_V2'+os.sep

    def getOutPath(self):
        return self.outPath

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


def main():
    # img_list = glob('casia2'+os.sep+'Au'+os.sep+'*')
    # cnnIns = cnn()
    # outDir = cnnIns.getOutPath() + 'Au_cropped'
    # if not os.path.exists(outDir):
    #     os.makedirs(outDir)
    # for imgPath in img_list:
    #     img = cnnIns.random_crop(imgPath, 128, 128)
    #     fName = outDir+os.sep+ntpath.basename(imgPath)
    #     fName = os.path.splitext(fName)[0]+'.jpg'
    #     img.save(fName, "JPEG")

    img_list = glob('casia2'+os.sep+'Tp'+os.sep+'*')
    cnnIns = cnn()
    outDir = cnnIns.getOutPath() + 'Tp_cropped'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    for imgPath in img_list:
        img = cnnIns.random_crop(imgPath, 128, 128)
        fName = outDir+os.sep+ntpath.basename(imgPath)
        fName = os.path.splitext(fName)[0]+'.jpg'
        img.save(fName, "JPEG")

main()
