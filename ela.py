from PIL import Image, ImageChops, ImageEnhance
from glob import glob
import ntpath
import os


class ela:

    def __init__(self, images):
        self.outPath = 'ela'+os.sep+'CASIA_V2'+os.sep+'compress'
        self.diffPath = 'ela'+os.sep+'CASIA_V2'+os.sep+'diff'
        self.images = images
        self.compressed = []
        self.set_compressed()

    def compress(self, qualityVal):
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
                extrema = diffImg.getextrema()
                max_diff = max([ex[1] for ex in extrema])
                scale = 255.0/max_diff

                diffImg = ImageEnhance.Brightness(diffImg).enhance(scale)
                diffImg.save(diffName,"JPEG")


def main():
    extract_dir = ['Au', 'Tp']
    for each_dir in extract_dir:
        img_list = glob('casia2'+os.sep+each_dir+os.sep+'*')
        elaIns = ela(img_list)
        elaIns.compress(90)
        print("performing ELA")
        elaIns.perform_ela()

main()

