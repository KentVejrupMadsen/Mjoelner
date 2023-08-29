from label_studio_ml.model \
    import LabelStudioMLBase

from model \
    import setup_model


class BackendModel(
    LabelStudioMLBase
):
    def __init__(
        self, 
        **kwargs
    ):
        super(BackendModel, self).__init__(**kwargs)
        
        self.model = setup_model(
            8
        )

    def train(self):
        pass