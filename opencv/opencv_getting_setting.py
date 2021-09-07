import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

(b, g, r) = image[299, 399]
print("Pixel at (399, 299) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

image[:299, :399] = (0, 255, 0)

cv2.imshow("Modified", image)

cv2.waitKey(0)