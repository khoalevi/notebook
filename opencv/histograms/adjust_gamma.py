import numpy as np
import argparse
import cv2


def adjust_gammar(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

for gamma in np.arange(0.0, 2.0, 0.5):
    if gamma == 1.0:
        continue

    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gammar(image, gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Images", np.hstack([image, adjusted]))
    cv2.waitKey(0)
