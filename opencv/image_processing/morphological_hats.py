import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default="phantom.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

cv2.imshow("Original", image)
cv2.imshow("Blackhat", blackhat)
cv2.imshow("Tophat", tophat)
cv2.waitKey(0)
