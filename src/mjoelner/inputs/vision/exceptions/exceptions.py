def raise_exception_for_no_input():
    raise IOError(
        'No input has been found.'
        +
        'allowed formats are streams (webcam, video capture, etc),'
        +
        'and videos (mp4, etc).'
    )


def raise_exception_for_video_format():
    raise IOError(
        'Video format is not supported.'
    )
