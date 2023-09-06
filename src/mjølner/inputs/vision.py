from cv2 \
    import VideoCapture

from cv2                    \
    import                  \
    COLOR_BGR2RGB,          \
    cvtColor,               \
    CAP_PROP_FRAME_HEIGHT,  \
    CAP_PROP_FRAME_WIDTH

from PIL \
    import Image


class Vision:
    def __init__(
        self,
        capture_device: str
    ):
        self.is_receiving_no_input: bool = False

        self.capture: VideoCapture = VideoCapture(
            capture_device
        )

        self.conversion_format = COLOR_BGR2RGB

    def __del__(self):
        self.capture.release()

    def get_conversion_format(self):
        return self.conversion_format

    def is_open(
            self
    ):
        return self.capture.isOpened()

    def raise_exception_for_no_input(self):
        raise IOError(
            'No input has been found.'
            +
            'allowed formats are streams (webcam, video capture, etc),'
            +
            'and videos (mp4, etc).'
        )

    def capture_view(
            self
    ) -> Image.Image:
        if self.has_no_input():
            self.raise_exception_for_no_input()

        returnable, frame = self.capture.read()

        if returnable:
            frame = cvtColor(
                frame,
                self.get_conversion_format()
            )

            return Image.fromarray(
                frame
            )
        else:
            self.flag_no_input()

    def has_no_input(
            self
    ) -> bool:
        return self.is_receiving_no_input

    def set_no_input(
            self,
            value: bool
    ) -> None:
        self.is_receiving_no_input = value

    def flag_no_input(
            self
    ) -> bool:
        self.set_no_input(
            True
        )

        return self.is_receiving_no_input

    def get_capture_width(self) -> int | None:
        if not self.is_capture_none():
            return int(
                self.get_capture().get(
                    CAP_PROP_FRAME_WIDTH
                )
            )

        return None

    def get_capture_height(self) -> int | None:
        if not self.is_capture_none():
            return int(
                self.get_capture().get(
                    CAP_PROP_FRAME_HEIGHT
                )
            )

        return None

    def is_capture_none(self) -> bool:
        return self.capture is None

    def get_capture(self):
        return self.capture

    def set_capture(
            self,
            value
    ) -> None:
        self.capture = value
