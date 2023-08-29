from keras_cv.models    \
    import              \
    YOLOV8Detector,     \
    YOLOV8Backbone

from labels             \
    import              \
    get_size_of_labels

def get_model():
    global model 

    if model is None:
        setup_model()

    return model

def setup_model() -> None:
    global model
    model = YOLOV8Detector(
        num_classes = get_size_of_labels(),
        bounding_box_format = 'xyxy',
        
        backbone=YOLOV8Backbone.from_preset(
            'yolo_v8_l_backbone_coco'
        ),

        fpn_depth=2
    )
