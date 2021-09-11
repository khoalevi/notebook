import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default="phantom.jpg", help="path to input image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
kernels = [(3, 3), (9, 9), (15, 15)]

for kernel in kernels:
    blurred = cv2.blur(image, kernel)
    cv2.imshow("Average {}".format(kernel), blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for kernel in kernels:
    blurred = cv2.GaussianBlur(image, kernel, 0)
    cv2.imshow("Gaussian {}".format(kernel), blurred)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for k in (3, 9, 15):
    blurred = cv2.medianBlur(image, k)
    cv2.imshow("Median {}".format(k), blurred)
    cv2.waitKey(0)