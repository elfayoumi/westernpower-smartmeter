import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from keras.models import model_from_json
import os
import sys
class MeterPredictor:
    def __init__(self, logger):
        try:
            self.logger = logger
            self.graph = tf.get_default_graph()
            json_file_path =os.path.join(os.path.dirname(__file__), "weights/model.json")
            weights_file_path =  os.path.join(os.path.dirname(__file__), "weights/weights2.60-0.05.hdf5")
            json_file = open(json_file_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(weights_file_path)
            self.model = loaded_model
        except:
            self.logger.debug("failed to initialise Meter Predictor")
            self.logger.debug(sys.exc_info())
            raise
    def predict(self, input_data):
        self.logger.debug("making a preditcion for sample with rowcount " + str (len(input_data[0])))
        results = self.model.predict(input_data)
        self.logger.debug("prediction succeeded")
        return results