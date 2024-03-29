import cv2


class ResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_LINEAR):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
