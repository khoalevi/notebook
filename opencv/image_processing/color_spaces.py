import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default="phantom.jpg",
                    help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("RGB", image)

for (name, channel) in zip(("B", "G", "R"), cv2.split(image)):
	cv2.imshow(name, channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

for (name, channel) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

for (name, channel) in zip(("L*", "a*", "b*"), cv2.split(lab)):
	cv2.imshow(name, channel)

cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)