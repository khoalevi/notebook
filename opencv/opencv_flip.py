import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

flipped = cv2.flip(image, 0)
cv2.imshow("Flipped vertically", flipped)

flipped = cv2.flip(image, 1)
cv2.imshow("Flipped horizontally", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped both axes", flipped)

cv2.waitKey(0)