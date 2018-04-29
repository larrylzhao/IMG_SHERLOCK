# USAGE
# python test_video.py -i resources/videos/Avengers.mp4 -m output/models/cnn_ela_patches.model


import argparse
import cv2


def perform_test_video():
    cap = cv2.VideoCapture(args["video"])

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    framepath = 'tmp/video_frames/frame.jpg'
    outpath = 'output/videos/test.mp4'

    out = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    while(cap.isOpened()):
        ret, frame = cap.read()

        # cv2.imwrite(framepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        out.write(frame)

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--video", required=True,
                help="path to input video")
ap.add_argument("-s", "--stepsize", default=128,
                help="step size for the sliding window (pixels)")

args = vars(ap.parse_args())

perform_test_video()
