import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, cnt):
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRation = w / float(h)
            shape ="square" if 0.95 >= aspectRation >= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"

        return shape