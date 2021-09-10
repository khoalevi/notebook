import argparse
import imutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, default="cat.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 degrees", rotated)

M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 degrees", rotated)

M = cv2.getRotationMatrix2D((10, 10), 15, 0.7)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by arbitrary point", rotated)

rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated by 180 degrees", rotated)

rotated = imutils.rotate_bound(image, -15)
cv2.imshow("Rotated without cropping", rotated)

cv2.waitKey(0)