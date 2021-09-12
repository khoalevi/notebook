import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                default="steve_jobs.jpeg", help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

(T, threshInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Simple thresholding", threshInv)
cv2.waitKey(0)

(T, threshInv) = cv2.threshold(blurred, 0,
                               255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Otsu thresholding", threshInv)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean adaptive thresholding", thresh)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
cv2.imshow("Gaussian adaptive thresholding", thresh)

cv2.waitKey(0)


