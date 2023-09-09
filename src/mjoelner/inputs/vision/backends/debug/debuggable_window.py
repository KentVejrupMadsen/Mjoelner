from mjoelner.inputs.vision.backends.debug \
    import RealtimeWindow


class DebuggableWindow(
    RealtimeWindow
):
    def __init__(
        self,
        exit_on_key: chr = '½'
    ):
        super().__init__()

        self.exit_on_keypress: int = ord(
            exit_on_key
        )

        self.exit: bool = False

    def set_exit_on_keypress(
        self,
        value: str | int
    ) -> None:
        if value is str:
            self.exit_on_keypress = ord(
                value
            )

        if value is int:
            self.exit_on_keypress = value

    def stream_input(
        self,
        frame
    ) -> None:
        self.show(
            frame
        )

        self.wait()

        if self.key_pressed == self.exit_on_keypress:
            self.flag_exit()

    def flag_exit(
        self
    ) -> None:
        self.set_exit(
            True
        )

    def set_exit(
        self,
        value: bool
    ) -> None:
        self.exit = value

    def get_exit(
        self
    ) -> bool:
        return self.exit
