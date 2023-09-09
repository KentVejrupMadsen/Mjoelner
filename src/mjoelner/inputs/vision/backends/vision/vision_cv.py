from cv2                    \
    import                  \
    CAP_PROP_FRAME_HEIGHT,  \
    CAP_PROP_FRAME_WIDTH,   \
    CAP_PROP_CONVERT_RGB,   \
    CAP_PROP_FRAME_COUNT,   \
    CAP_PROP_FPS

from cv2 \
    import VideoCapture

from cv2            \
    import          \
    COLOR_BGR2RGB,  \
    cvtColor,       \
    COLOR_RGB2GRAY, \
    COLOR_RGB2BGR


class VisionCV:
    def __init__(
        self,
        capture_device: str
    ):
        self.conversion_format: int = COLOR_BGR2RGB

        self.capture: VideoCapture = VideoCapture(
            capture_device
        )

    def __del__(self):
        self.capture.release()

