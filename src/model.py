from os.path \
    import isdir, isfile, join

from keras.models \
    import load_model

from os \
    import environ

from keras.models \
    import Model

from keras_cv.models    \
    import              \
    YOLOV8Detector,     \
    YOLOV8Backbone


default_model_location: str | None = None
default_environment_variable_location: str = 'mjoelner_model_location'
default_environment_variable_location = default_environment_variable_location.upper()


if default_environment_variable_location in environ:
    default_model_location = environ[
        default_environment_variable_location
    ]


def get_model_location() -> str | None:
    global default_model_location
    return default_model_location


def is_default_location_set() -> bool:
    global default_model_location
    return not(
        default_model_location is None
    )

def set_model_location(
    value: str
) -> None:
    global default_model_location
    default_model_location = value

class MjoelnerModelFramework:
    def __init__(
        self,
        labels: list,
        model: Model | None = None,
    ):
        self.model = model
        self.labels: list = labels
        self.load_from_location()
    
    def get_model(self) -> Model:
        return self.model
    
    def set_model(
        self, 
        model: Model
    ) -> None:
        self.model = model

    def load_from_location(self):
        if is_default_location_set():
            if isdir(
                get_model_location()
            ):
                self.set_model(
                    load_model(
                        get_model_location()
                    )
                )
        else:
            self.setup_from_scratch()
    
    def setup_from_scratch(self):
        self.model = YOLOV8Detector(
            num_classes=len(self.labels),
            bounding_box_format = 'xyxy',
            backbone=YOLOV8Backbone.from_preset(
                'yolo_v8_l_backbone_coco'
            ),
            fpn_depth=2
        )


