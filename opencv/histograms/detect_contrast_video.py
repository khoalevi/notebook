from skimage.exposure import is_low_contrast
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="contrast.mp4",
                help="path to the video")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
                help="threshold for contrast")
args = vars(ap.parse_args())


vs = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.putText(frame, text, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    output = np.dstack([edged] * 3)
    output = np.hstack([frame, output])

    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
	    break
