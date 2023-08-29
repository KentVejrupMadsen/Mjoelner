model_configuration: dict = {
    'labels': [
        'actor',
        'decal',
        'entity',
        'guide',
        'object',
        'overlay',
        'segment',
        'surface'
    ],

    'size_of_labels': 8
}

def get_configuration() -> dict:
    global model_configuration
    return model_configuration

def get_size_of_labels() -> int:
    return get_configuration()[
        'size_of_labels'
    ]