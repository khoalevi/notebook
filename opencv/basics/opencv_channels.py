import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(B, G, R) = cv2.split(image)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)

cv2.waitKey(0)

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

cv2.waitKey(0)
cv2.destroyAllWindows()

zeros = np.zeros(image.shape[:2], dtype="uint8")
red = cv2.merge([zeros, zeros, R])
green = cv2.merge([zeros, G, zeros])
blue = cv2.merge([B, zeros, zeros])

cv2.imshow("Red", red)
cv2.imshow("Green", green)
cv2.imshow("Blue", blue)

cv2.waitKey(0)