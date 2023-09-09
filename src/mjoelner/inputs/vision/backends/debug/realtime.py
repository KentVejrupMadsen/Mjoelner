from cv2    \
    import  \
    imshow, \
    waitKey


class RealtimeWindow:
    def __init__(
        self
    ):
        self.time_to_wait_on_key: int = 5
        self.title: str = 'Debugging, Vision'

        self.key_pressed: int | None = None

    def stream_input(
        self,
        frame
    ) -> None:
        self.show(
            frame
        )
        self.wait()

    def get_key_pressed(
        self
    ) -> int:
        return self.key_pressed

    def set_key_pressed(
        self,
        value: int
    ) -> None:
        self.key_pressed = value

    def show(
        self,
        frame
    ):
        imshow(
            self.title,
            frame
        )

    def wait(
        self
    ):
        key = waitKey(
            self.time_to_wait_on_key
        )

        self.set_key_pressed(
            key
        )
