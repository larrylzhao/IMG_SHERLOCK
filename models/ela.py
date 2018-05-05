from PIL import Image, ImageChops, ImageEnhance
from glob import glob
import ntpath
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout

class ela:

    def __init__(self, images, outPath):
        self.outPath = outPath+'/compress'
        self.diffPath = outPath+'/diff'
        self.images = images
        self.compressed = []
        self.set_compressed()

        if not os.path.exists(self.outPath):
            os.makedirs(self.outPath)
        if not os.path.exists(self.diffPath):
            os.makedirs(self.diffPath)

    def compress(self, qualityVal):
        print("compressing images")
        for img in self.images:
            if not (img.endswith('db')):
                image = Image.open(img)
                filename = self.outPath+os.sep+ntpath.basename(img)
                image.save(os.path.splitext(filename)[0]+'.jpg', "JPEG", quality=qualityVal)
                self.set_compressed()

    def get_images(self):
        return self.images

    def get_compressed(self):
        return self.compressed

    def set_compressed(self):
        # print("setting compressed")
        for img in self.images:
            filename = self.outPath+os.sep+ntpath.basename(img)
            self.compressed.append(filename)

    def perform_ela(self):
        for img in self.images:
            if not (img.endswith('db')):
                image = Image.open(img)
                filename = self.outPath+os.sep+ntpath.basename(img)
                image2 = Image.open(os.path.splitext(filename)[0]+'.jpg')
                print(img, filename)
                diffImg = ImageChops.difference(image, image2)
                diffName = self.diffPath+os.sep+ntpath.basename(img)
                diffName = os.path.splitext(diffName)[0]+'.jpg'
                # extrema = diffImg.getextrema()
                # max_diff = max([ex[1] for ex in extrema])
                # scale = 255.0/max_diff
                #
                # diffImg = ImageEnhance.Brightness(diffImg).enhance(scale)
                diffImg.save(diffName,"JPEG")

    @staticmethod
    def build_model(dim, classes):
        # MLP for ELA data
        model = Sequential()

        model.add(Dense(1024, input_dim=dim, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='sigmoid'))

        # return the constructed network architecture
        return model


def main():

    # img_list = glob('resources/casia2/Au_128_128/*')
    # elaIns = ela(img_list, 'output/ela/CASIA_V2/Au')
    # elaIns.compress(90)
    # print("performing ELA")
    # elaIns.perform_ela()

    img_list = glob('resources/casia2/Tp_auto_patches/*')
    elaIns = ela(img_list, 'output/ela/CASIA_V2_AUTO_PATCHES/Tp')
    elaIns.compress(90)
    print("performing ELA")
    elaIns.perform_ela()

# main()

