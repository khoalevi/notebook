import argparse
import imutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

ratio = 150 / image.shape[1]
dim = (150, int(image.shape[0] * ratio))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

resized = imutils.resize(image, width=200)
cv2.imshow("Resized via imutils", resized)

cv2.waitKey(0)

methods = [
    ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
    ("cv2.INTER_LINEAR", cv2.INTER_LINEAR),
    ("cv2.INTER_AREA", cv2.INTER_AREA),
    ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
    ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)
]

for (name, method) in methods:
    resized = imutils.resize(image, image.shape[1] * 2, inter=method)
    cv2.imshow("Method: {}".format(name), resized)
    cv2.waitKey(0)