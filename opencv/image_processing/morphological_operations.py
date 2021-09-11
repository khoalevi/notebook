import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", nargs='?',
                    choices=['text', 'noise'], type=str, default='text')
args = vars(parser.parse_args())

src = 'text.png' if args["mode"] == 'text' else 'text-with-noise.png'

image = cv2.imread(src)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

for i in range(3):
    eroded = cv2.erode(gray.copy(), None, iterations=i+1)
    cv2.imshow("Eroded {} times".format(i + 1), eroded)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for i in range(3):
    eroded = cv2.dilate(gray.copy(), None, iterations=i+1)
    cv2.imshow("Dilated {} times".format(i + 1), eroded)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (5, 5), (7, 7)]

for size in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening: ({}. {})".format(size[0], size[1]), opening)
    

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for size in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing: ({}. {})".format(size[0], size[1]), closing)
    cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.imshow("Original", image)

for size in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    closing = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Gradient: ({}. {})".format(size[0], size[1]), closing)
    cv2.waitKey(0)