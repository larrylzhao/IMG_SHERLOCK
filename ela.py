from PIL import Image
from glob import glob
import ntpath
import os

class ela:

    def __init__(self, images):
        self.images = images

    def compress(self, outPath, qualityVal):
        for img in self.images:
            image = Image.open(img)
            image.save(outPath+os.sep+ntpath.basename(img),"JPEG", quality=qualityVal)



def main():
    extract_dir = ['Au', 'Sp']
    for each_dir in extract_dir:
        img_list = glob('casia1'+os.sep+each_dir+os.sep+'*.jpg')
        outPath = 'ela'+os.sep+'compress'
        elaIns = ela(img_list)
        elaIns.compress(outPath, 90)

main()

