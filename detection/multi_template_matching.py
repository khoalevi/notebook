from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input image")
ap.add_argument("-t", "--template", type=str, required=True,
                help="path to template image")
ap.add_argument("-b", "--threshold", type=float, default=0.8,
                help="threshold for matching")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]

cv2.imshow("Image", image)
cv2.imshow("Template", template)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

(yCoords, xCoords) = np.where(result >= args["threshold"])
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

for (x, y) in zip(xCoords, yCoords):
    cv2.rectangle(clone, (x, y), (x + tW, y + tH), (0, 255, 0), 3)

cv2.imshow("Before NMS", clone)
cv2.waitKey(0)

rects = []

for (x, y) in zip(xCoords, yCoords):
    rects.append((x, y, x + tW, y + tH))

pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)

cv2.imshow("After NMS", image)
cv2.waitKey(0)