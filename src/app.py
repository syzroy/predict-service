from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('LSTM.h5')

import tensorflow as tf

def load_model():
	global cnn
	cnn = InceptionResNetV2(include_top=False, weights='imagenet')
	global graph
	graph = tf.get_default_graph()

from flask import Flask, request, make_response, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
  with graph.as_default():
    data = request.data
    img = pickle.loads(data)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn.predict(x)
    features = features[0]
    li = []
    li.append(features)
    li.append(features)
    features = [li]
    features = np.array(features)
    features = features.reshape(
        features.shape[0], features.shape[1],
        features.shape[2] * features.shape[3] * features.shape[4])

    result = model.predict(features)
  return jsonify({'result': result.tolist() })


if __name__ == "__main__":
    app.run(host='0.0.0.0')
