from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt

leaf_disease_model = load_model('saved_model.h5')
label_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
               'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust',
               'Corn Northern Leaf Blight', 'Corn healthy',
               'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot',
               'Peach healthy', 'Pepper bell Bacterial spot',
               'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy',
               'Strawberry Leaf scorch', 'Strawberry healthy',
               'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
               'Tomato Septoria leaf spot',
               'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
               'Tomato healthy']

app = Flask(__name__)


@app.route("/", methods=['POST'])
def classify_image():
    img = request.files['img']
    img_path = save_temp_file(img)

    img = image.load_img(img_path, target_size=(150, 150))  # Resize the image to match VGG16's input shape
    img = image.img_to_array(img)
    img = preprocess_input(img)  # Preprocess the image according to VGG16's requirements

    os.remove(img_path)  # Remove the temporary file

    img = np.expand_dims(img, axis=0)  # Add an extra dimension as the batch dimension

    predictions = leaf_disease_model.predict(img)

    label_index = np.argmax(predictions)
    label_name = label_names[label_index]
    accuracy = predictions[0][label_index] * 100

    # Generate a bar graph of the prediction probabilities
    y_pos = np.arange(len(label_names))
    plt.bar(y_pos, predictions[0])
    plt.xticks(y_pos, label_names, rotation='vertical')
    plt.ylabel('Probability')
    plt.xlabel('Label')
    plt.title('Disease Classification')

    # Save the graph to a temporary file
    _, graph_path = tempfile.mkstemp('.png')
    plt.savefig(graph_path)
    plt.close()

    return jsonify({
        "Label Name": label_name,
        "Accuracy": accuracy,
        "Graph": graph_path
    })


def save_temp_file(file):
    _, temp_path = tempfile.mkstemp()  # Create a temporary file
    file.save(temp_path)  # Save the FileStorage object to the temporary file
    return temp_path


if __name__ == "__main__":
    app.run(debug=True)