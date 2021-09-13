import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-m", "--mode", nargs='?',
                    choices=["sobel", "scharr"], type=str, default="sobel")
args = vars(parser.parse_args())

mode = args["mode"]

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

ksize = -1 if mode == "scharr" else 3

gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.imshow("{} X".format(mode), gX)
cv2.imshow("{} Y".format(mode), gY)
cv2.imshow("{} combined".format(mode), combined)
cv2.waitKey(0)
