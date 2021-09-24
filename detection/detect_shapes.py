from levi.shape_detector import ShapeDetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
aspectRatio = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

for cnt in cnts:
    M = cv2.moments(cnt)
    cntX = int((M["m10"] / M["m00"]) * aspectRatio)
    cntY = int((M["m01"] / M["m00"]) * aspectRatio)
    shapeName = sd.detect(cnt)

    cnt = cnt.astype("float")
    cnt = cnt * aspectRatio
    cnt = cnt.astype("int")

    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(image, shapeName, (cntX, cntY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)