import numpy as np
import cv2
from numpy.core.fromnumeric import size

canvas = np.zeros((300, 300, 3), dtype="uint8")

green = (0, 255, 0)
cv2.line(canvas, (300, 0), (0, 300), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

red = (0, 0, 255)
cv2.line(canvas, (0, 0), (300, 300), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

blue = (255, 0, 0)
cv2.rectangle(canvas, (50, 50), (250, 250), blue)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (25, 75), (275, 225), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (125, 125), (175, 75), green, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

canvas = np.zeros((300, 300, 3), dtype="uint8")
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

for radius in range(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), radius, white)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

canvas = np.zeros((300, 300, 3), dtype="uint8")

for i in range(0, 10):
    radius = np.random.randint(5, high=200)
    color = np.random.randint(0, high=256, size=(3,))
    position = np.random.randint(0, high=300, size=(2,))
    cv2.circle(canvas, position, radius, color, -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)