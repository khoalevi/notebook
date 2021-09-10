import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

added = cv2.add(np.uint8([200]), np.uint8([100]))
subtracted = cv2.subtract(np.uint8([50]), np.uint8([100]))
print("cv2: 200 + 100 = {}".format(added))
print("cv2: 50 - 100 = {}".format(subtracted))

added = np.add(np.uint8([200]), np.uint8([100]))
subtracted = np.subtract(np.uint8([50]), np.uint8([100]))
print("np: 200 + 100 = {}".format(added))
print("np: 50 - 100 = {}".format(subtracted))

M = np.ones(image.shape, dtype="uint8") * 50

lighter = cv2.add(image, M)
cv2.imshow("Lighter", lighter)

darker = cv2.subtract(image, M)
cv2.imshow("Darker", darker)

cv2.waitKey(0)
