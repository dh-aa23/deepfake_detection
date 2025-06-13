from tensorflow.keras.models import load_model
# from tensorflow import keras

class Predictor:
    def __init__(self,model_path):
        self.model_path=model_path
    def get_prediction(self,features,mask):
        model=load_model(self.model_path)
        return model.predict([features,mask])


