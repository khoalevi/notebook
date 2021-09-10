import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
(h, w, c) = image.shape[:3]

print("width: {} pixels".format(w))
print("height: {} pixels".format(h))
print("channels: {}".format(c))

cv2.imshow("Image", image)
cv2.waitKey(0)