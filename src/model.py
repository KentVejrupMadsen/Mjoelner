from keras_cv.models    \
    import              \
    YOLOV8Detector,     \
    YOLOV8Backbone

def get_model():
    global model 

    return model

def setup_model(
    size_of_categories: int
) -> None:
    global model

    model = YOLOV8Detector(
        num_classes = size_of_categories,
        bounding_box_format = 'xyxy',
        
        backbone=YOLOV8Backbone.from_preset(
            'yolo_v8_l_backbone_coco'
        ),

        fpn_depth=4
    )
