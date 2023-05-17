from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K

app = Flask(__name__)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Load your model and weights
model = tf.keras.models.load_model("./models/best_weights.hdf5", custom_objects={'f1_score': f1_score})
@app.route("/very_mild_demented")
def very_mild_demented():
    return render_template("very_mild_demented.html")

@app.route("/mild_demented")
def mild_demented():
    return render_template("mild_demented.html")

@app.route("/moderate_demented")
def moderate_demented():
    return render_template("moderate_demented.html")

@app.route("/non_demented")
def non_demented():
    return render_template("non_demented.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image from the user
        img = Image.open(request.files["image"].stream).convert("RGB")
    
        # Resize the image to match the model input size
        img = img.resize((224, 224))
        img_arr = np.array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Make a prediction using the model
        predictions = model.predict(img_arr)
        class_probs = predictions[0] * 100
        result = {
            "class_probabilities": {
                "mild_demented": float(class_probs[0]),
                "moderate_demented": float(class_probs[1]),
                "non_demented":float(class_probs[2]),
                "very_mild_demented": float(class_probs[3]),
            }
        }

        return jsonify(result)
    except Exception as e:
        return str(e), 500  # Return the error message as a response

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
