from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow import keras
IMG_SIZE=299
class Feature_extractor:
    def __init__(self):
        pass

    def build_feature_extractor(self):
        extractor = InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling='avg',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)
        out = extractor(preprocessed)

        return keras.Model(inputs, out, name='extractor')
    
    def get_extractor(self):
        feature_extractor= self.build_feature_extractor()
        return feature_extractor
