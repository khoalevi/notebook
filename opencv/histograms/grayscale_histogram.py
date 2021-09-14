import matplotlib.pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

plt.figure()
plt.title("Grayscale histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0, 256])

hist /= hist.sum()

plt.figure()
plt.title("Grayscale histogram (normalized)")
plt.xlabel("Bins")
plt.ylabel("% of pixels")
plt.plot(hist)
plt.xlim([0, 256])

plt.show()