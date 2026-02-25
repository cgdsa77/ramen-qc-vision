import cv2


class VideoReader:
    def __init__(self, path: str, fps: int = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.cap.release()
