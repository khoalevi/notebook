from skimage.exposure.exposure import is_low_contrast
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                type=str, help="path to the image")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
                help="threshold for contrast")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

text = "Low contrast: No"
color = (0, 255, 0)

if is_low_contrast(gray, fraction_threshold=args["thresh"]):
    text = "Low contrast: Yes"
    color = (0, 0, 255)
else:
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Original", image)
cv2.imshow("Edged", edged)

cv2.waitKey(0)
