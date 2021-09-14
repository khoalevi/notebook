import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


def plot_histogram(image, title, mask=None):
    channels = cv2.split(image)
    colors = ("b", "g", "r")

    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="cat.jpg",
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
plot_histogram(image, "Histogram for original image")

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (120, 80), (475, 425), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked", masked)

plot_histogram(image, "Histogram for masked image", mask=mask)

plt.show()