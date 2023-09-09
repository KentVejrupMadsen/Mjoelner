from mjoelner.inputs.vision \
    import Vision

location_of_test_video: str = 'C:\\Users\\Kentv\\Videos\\Halo\\CE\\Halo  The Master Chief Collection 2023.09.08 - 17.23.42.01.mp4'


def create_vision():
    global location_of_test_video
    vision = Vision(
        location_of_test_video
    )

    return vision


def test_of_vision():
    vision = create_vision()
    vision.is_debugging = True

    while vision.is_input_stream_open():
        vision.capture_view()

        if not vision.is_debugging:
            break

        if vision.get_flag_stop():
            break

    assert True


def test_section():
    vision = create_vision()
    vision.capture_view()

    sections = vision.sections(
        2,
        2
    )

    assert len(sections) > 0
