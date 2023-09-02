from os import listdir
from os.path import isdir, join

from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow
import keras

import keras_cv
from keras_cv import bounding_box, visualization
from random import SystemRandom

import wandb
from wandb import init
from wandb.integration.keras import WandbMetricsLogger


split_at: float = 0.35
batch_size: int = 24

learning_rate: float = 0.002

epoch: int = 1
global_clip_at: float = 10.0

boundary_format: str = 'xyxy'

labels: list = [
    'Dog',
    'Dog tag',
    'Collar'
]

labels_mapped: dict = dict(
    zip(
        range(len(labels)), 
        labels
    )
)

dataset_paths: dict = dict(
    {
        'images': 'D:\\DataSet\\Tracking Madsen\\Images',
        'annotations': 'D:\\DataSet\\Tracking Madsen\\Annotations'
    }
)


target_size_width: int = 640 
target_size_height: int = 640


###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

init(
    project='YOLO', 
    entity='designermadsen',
    save_code=True
)

###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################

xml_files = sorted(
    [
        join(dataset_paths['annotations'], file_name)
        for file_name in listdir(dataset_paths['annotations'])
        if file_name.lower().endswith(".xml")
    ]
)

size_of_annotations = len(xml_files)

image_files = sorted(
    [
        join(dataset_paths['images'], file_name)
        for file_name in listdir(dataset_paths['images'])
        if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg") or file_name.lower().endswith(".png")
    ]
)

def parse_xml(
    xml_file
):
    global dataset_paths, labels_mapped
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = join(dataset_paths['images'], image_name)

    boxes = []
    classes = []

    for object in root.iter('object'):
        cls = object.find("name").text
        classes.append(cls)

        bounderies = object.find("bndbox")
        xmin: float = float(bounderies.find('xmin').text)
        ymin: float = float(bounderies.find('ymin').text)

        xmax: float = float(bounderies.find('xmax').text)
        ymax: float = float(bounderies.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
    
    label_ids: list = [
        list(labels_mapped.keys())[list(labels_mapped.values()).index(cls)]
        for cls in classes
    ]
    
    return image_path, boxes, label_ids

image_paths = []
boundaries = []
classes = []

for file in tqdm(
    xml_files
):
    image_path, boxes, label_ids = parse_xml(
        file
    )
    
    image_paths.append(
        image_path
    )
    
    boundaries.append(
        boxes
    )
    
    classes.append(
        label_ids
    )

from tensorflow.ragged import constant

image_paths = constant(image_paths)
classes = constant(classes)
boundaries = constant(boundaries)

dataset = tensorflow.data.Dataset.from_tensor_slices((
    image_paths,
    classes,
    boundaries
))

def load_image(image_path):
    image = tensorflow.io.read_file(image_path)
    image = tensorflow.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(
    image_path, 
    classes, 
    bounderies
):
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tensorflow.cast(
            classes, 
            dtype=tensorflow.float32
        ),
        "boxes": bounderies
    }

    return {
        "images": tensorflow.cast(
            image, 
            dtype=tensorflow.float32
        ), 
        "bounding_boxes": bounding_boxes
    }

resize = keras.Sequential(
    layers=[
        keras_cv.layers.Resizing(
            width=target_size_width, 
            height=target_size_height, 
            pad_to_aspect_ratio=True, 
            bounding_box_format=boundary_format
        )
    ]
)

number_of_validation: int = int(size_of_annotations * split_at)

validation_data = dataset.take(
    number_of_validation
)

training_data = dataset.skip(
    number_of_validation
)

training_data = training_data.map(
    load_dataset, 
    num_parallel_calls=tensorflow.data.AUTOTUNE
)

training_data = training_data.shuffle(batch_size * 4)

training_data = training_data.ragged_batch(
    batch_size, 
    drop_remainder=True
)

training_data = training_data.map(
    resize, 
    num_parallel_calls=tensorflow.data.AUTOTUNE
)


validation_data = validation_data.map(
    load_dataset, 
    num_parallel_calls=tensorflow.data.AUTOTUNE
)

validation_data = validation_data.shuffle(batch_size * 4)

validation_data = validation_data.ragged_batch(
    batch_size, 
    drop_remainder=True
)

validation_data = validation_data.map(
    resize, 
    num_parallel_calls=tensorflow.data.AUTOTUNE
)

def visualise_dataset(
        dataset_as_input, 
        value_range, 
        rows, 
        cols, 
        bounding_box_format
):
    global labels_mapped
    dataset_as_input = next(iter(dataset_as_input.take(1)))
    images, bounding_boxes = dataset_as_input["images"], dataset_as_input["bounding_boxes"]
    
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.5,
        bounding_box_format=bounding_box_format,
        class_mapping=labels_mapped
    )

visualise_dataset_before_run: bool = False

if visualise_dataset_before_run:
    visualise_dataset(
        training_data, 
        bounding_box_format=boundary_format, 
        value_range=(0, 255), 
        rows=2, 
        cols=2
    )

    visualise_dataset(
        validation_data, 
        bounding_box_format=boundary_format, 
        value_range=(0, 255), 
        rows=2, 
        cols=2
    )

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

training_data = training_data.map(dict_to_tuple, num_parallel_calls=tensorflow.data.AUTOTUNE)
training_data = training_data.prefetch(tensorflow.data.AUTOTUNE)

validation_data = validation_data.map(dict_to_tuple, num_parallel_calls=tensorflow.data.AUTOTUNE)
validation_data = validation_data.prefetch(tensorflow.data.AUTOTUNE)


###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
###############################################################################################################################################################################
from keras.layers import InputLayer
from keras import Model

# Model Creation
backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_m_backbone_coco" 
)

yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(labels_mapped),
    bounding_box_format=boundary_format,
    backbone=backbone,
    fpn_depth=2
)

optimizer = tensorflow.keras.optimizers.Adam(
    learning_rate=learning_rate,
    global_clipnorm=global_clip_at,
)

yolo.compile(
    optimizer=optimizer, 
    classification_loss="binary_crossentropy", 
    box_loss="ciou"
)


yolo.fit(
    training_data,
    validation_data=validation_data,
    epochs=epoch,
    use_multiprocessing=True,
    workers=4,
    callbacks=[
        WandbMetricsLogger()
    ]
)


artifact = wandb.Artifact(
    name='yolo_model', 
    type='model'
)

artifact.add_dir(
    local_path='D:\\Model\\yolo'
)

wandb.log_artifact(
    artifact
)