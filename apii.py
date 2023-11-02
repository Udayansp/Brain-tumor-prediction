from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet_v2 import preprocess_input
from keras.applications.resnet50 import ResNet50
import io
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your custom ResNet50V2-based model
model = tf.keras.models.load_model("Brain-tumor-prediction/brain_tumor_classification.h5")

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    if imagefile:
        try:
            image = Image.open(imagefile)
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
            return render_template('index.html', prediction=class_names[predicted_class])
        except Exception as e:
            # Log the error for debugging
            print(f"Error: {str(e)}")
            return render_template('index.html', prediction='Error occurred during prediction')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=8080)







