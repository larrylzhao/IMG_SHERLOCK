import cv2
import ntpath
import re
from PIL import Image
import uuid
from glob import glob


def random_crop(imgPath, x, y, height, width):
    img = Image.open(imgPath)

    # check if the image dimensions are at least as big as the crop size
    imgWidth, imgHeight = img.size
    if (imgWidth < width) or (imgHeight < height):
        raise Exception('Image size is too small for the crop size')
    else:
        img = img.crop((x, y, x + width, y + height))
        return img


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


badimgs = {
    "cha10122": True,
    "sec20007": True,
    "nat10130": True,
    "sec00098": True,
    "sec00093": True,
    "ani10191": True
}


def extract_patch(tpimgpath):

    # print(tpimgpath)
    filename = ntpath.basename(tpimgpath)
    # print(filename)
    m = re.search('(\w{3})(\d{5})', filename)
    auimgpath = 'resources/casia2/Au/Au_' + m.group(1) + '_' + m.group(2)
    # if m.group(1)+m.group(2) in badimgs:
    #     return
    # print(auimgpath)
    tpimage = cv2.imread(tpimgpath)
    auimage = cv2.imread(auimgpath + '.jpg')
    if auimage is None:
        auimage = cv2.imread(auimgpath + '.bmp')
        if auimage is None:
            # raise Exception("image cannot be read or does not exist. " + tpimgpath)
            # exit()
            print("image couldn't be read or does not exist. " + tpimgpath)
            return
    if auimage.shape[0] != tpimage.shape[0] \
            and auimage.shape[0] != tpimage.shape[0]:
        print("image dimensions mismatched. " + tpimgpath)
        return
    (winW, winH) = (128, 128)
    threshold = 10
    for (x, y, window) in sliding_window(tpimage, stepSize=int(128), windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        diffcnt = 0
        for xval in range(x, x+winH):
            for yval in range(y, y+winW):
                tpchannels = tpimage[yval][xval]
                auchannels = auimage[yval][xval]
                if (abs(int(tpchannels[0]) - int(auchannels[0])) > threshold) \
                        or (abs(int(tpchannels[1]) - int(auchannels[1])) > threshold)\
                        or (abs(int(tpchannels[2]) - int(auchannels[2])) > threshold):
                    diffcnt += 1
                    # print(tpchannels)
                    # print(auchannels)
                    # print(abs(int(tpchannels[0]) - int(auchannels[0])))
                    # print("\n")

        pctdiff = float(diffcnt/(winH*winW)*100)

        # print(diffcnt)
        # print(pctdiff)

        if pctdiff > 5 and pctdiff < 70:
            filename = 'resources/casia2/Tp_auto_patches/' + uuid.uuid4().hex[:6].upper() + '.jpg'
            cv2.imwrite(filename, window, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def main():
    img_list = glob('resources/casia2/Tp/*')
    for imgPath in img_list:
        print(imgPath)
        extract_patch(imgPath)

main()
# extract_patch('resources/casia2/Tp/Tp_D_CNN_M_N_ani00057_ani00055_11149.jpg')
