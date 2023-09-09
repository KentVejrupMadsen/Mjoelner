from mjoelner.inputs.vision \
    import Vision

location_of_test_video: str = 'C:\\Users\\Kentv\\Videos\\Halo  The Master Chief Collection 2023.09.05 - 19.55.02.03.mp4'


def create_vision():
    global location_of_test_video
    vision = Vision(
        location_of_test_video
    )

    return vision


def test_of_vision():
    vision = create_vision()

    assert True


def test_section():
    vision = create_vision()

