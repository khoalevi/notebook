import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--coin", nargs='?', choices=['1', '2'], default='1')
args = vars(parser.parse_args())

src = 'coin1.jpg' if args["coin"] == '1' else 'coin2.jpg'

image = cv2.imread(src)
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Otsu's thresholding value: {}".format(T), threshInv)

masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("Masked", masked)
cv2.waitKey(0)
