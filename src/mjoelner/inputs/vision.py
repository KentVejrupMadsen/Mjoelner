from cv2 \
    import VideoCapture

from cv2                    \
    import                  \
    COLOR_BGR2RGB,          \
    cvtColor,               \
    CAP_PROP_FRAME_HEIGHT,  \
    CAP_PROP_FRAME_WIDTH,   \
    CAP_PROP_CONVERT_RGB,   \
    CAP_PROP_FRAME_COUNT,   \
    CAP_PROP_FPS

from PIL \
    import Image

from mjoelner.inputs.exceptions      \
    import                          \
    raise_exception_for_no_input,   \
    raise_exception_for_video_format


from mjoelner.inputs.support     \
    import                      \
    is_file_a_video,            \
    is_video_format_supported


class Vision:
    def __init__(
        self,
        capture_device: str
    ):
        self.is_receiving_no_input: bool = False

        if is_file_a_video(
                capture_device
        ):
            if not is_video_format_supported(
                capture_device
            ):
                raise_exception_for_video_format()

        self.capture: VideoCapture = VideoCapture(
            capture_device
        )

        self.conversion_format: int = COLOR_BGR2RGB
        self.is_debugging: bool = False

        self.current_frame = None

    def __del__(
            self
    ):
        self.capture.release()

    def get_current_frame(self) -> None | Image.Image:
        return self.current_frame

    def set_current_frame(
            self,
            image
    ) -> None:
        self.current_frame = image

    def get_conversion_format(
            self
    ):
        return self.conversion_format

    def is_input_stream_open(
        self
    ):
        return not self.is_receiving_no_input

    def buffer_is_empty(self) -> bool:
        return self.current_frame is None

    def sections(
        self,
        x: int,
        y: int
    ):
        if self.buffer_is_empty():
            self.capture_view()

        width, height = self.retrieve_width_and_height()

        sample_width = int(
            width/x
        )

        sample_height = int(
            height/y
        )

        returnables: list = list()

        for iw in range(x):
            next_iw = iw + 1

            for ih in range(y):
                next_ih = ih + 1

                top, bottom = (
                    int(next_ih * sample_height),
                    int(ih * sample_height)
                )

                right, left = (
                    int(next_iw * sample_width),
                    int(iw * sample_width)
                )

                returnables.append(
                    self.get_current_frame().crop(
                        (left, bottom, right, top)
                    )
                )

        return returnables

    def capture_view(
         self
    ) -> None:
        if self.has_no_input():
            raise_exception_for_no_input()

        if not self.capture.isOpened():
            self.flag_no_input()

        is_returnable, frame = self.capture.read()

        if is_returnable:
            self.push_raw_input_frame_to_debug_window(
                frame
            )

            frame = cvtColor(
                frame,
                self.get_conversion_format()
            )

            self.set_current_frame(
                Image.fromarray(
                    frame
                )
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

    def retrieve_width(
            self
    ) -> int | None:
        if not self.is_capture_none():
            return int(
                self.get_capture().get(
                    CAP_PROP_FRAME_WIDTH
                )
            )

        return None

    def retrieve_height(
            self
    ) -> int | None:
        if not self.is_capture_none():
            return int(
                self.get_capture().get(
                    CAP_PROP_FRAME_HEIGHT
                )
            )

        return None

    def retrieve_width_and_height(
            self
    ) -> tuple | None:
        if not self.is_capture_none():
            return (
                self.retrieve_width(),
                self.retrieve_height()
            )

        return None

    def is_capture_none(
            self
    ) -> bool:
        return self.capture is None

    def get_capture(
            self
    ) -> VideoCapture | None:
        return self.capture

    def set_capture(
            self,
            value: VideoCapture
    ) -> None:
        self.capture = value

    def retrieve_need_conversion_to_rgb_colorspace(
            self
    ) -> bool | None:
        if not self.is_capture_none():
            return bool(
                self.get_capture().get(
                    CAP_PROP_CONVERT_RGB
                )
            )

        return None

    def retrieve_total_number_of_frames(
            self
    ) -> int | None:
        if not self.is_capture_none():
            return int(
                self.get_capture().get(
                    CAP_PROP_FRAME_COUNT
                )
            )

        return None

    def retrieve_frames_per_second(
            self
    ) -> float | None:
        if not self.is_capture_none():
            return float(
                self.get_capture().get(
                    CAP_PROP_FPS
                )
            )

        return None

    def push_raw_input_frame_to_debug_window(
            self,
            frame
    ):
        if self.is_debugging:
            pass
