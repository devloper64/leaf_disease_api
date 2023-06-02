import time
import base64

from flask import Flask, request, jsonify, send_file
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import shutil
import io

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

# Dictionary to store symptoms and treatment information for each disease
disease_info = {
    'Apple scab': {
        'symptoms': 'The symptoms of Apple scab include dark, scaly lesions on the leaves, fruit, and twigs of apple trees. These lesions may also have a velvety or cork-like appearance. In severe cases, the fruit may become deformed or cracked.',
        'treatment': 'To treat Apple scab, it is recommended to remove and destroy infected plant material, improve air circulation around the tree, and apply fungicides according to the appropriate schedule.',
    },
    'Apple Black rot': {
        'symptoms': 'Apple Black rot causes circular, sunken lesions on the fruit. The lesions are usually brown or black, and they may have concentric rings. Infected fruit often shrivel and mummify.',
        'treatment': 'To control Apple Black rot, it is important to remove and destroy infected fruit and prune out any cankers. Fungicides can also be used as a preventative measure.',
    },
    'Apple Cedar apple rust': {
        'symptoms': 'Cedar apple rust causes orange or rust-colored spots on the leaves, fruit, and twigs of apple trees. The spots may have a raised, pustule-like appearance. In advanced stages, galls may form on the twigs.',
        'treatment': 'The treatment for Cedar apple rust involves pruning out infected twigs and applying fungicides. It is also important to control the alternate hosts, such as cedar or juniper trees.',
    },
    'Apple healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Cherry Powdery mildew': {
        'symptoms': 'Powdery mildew causes a powdery white coating on the leaves, shoots, and fruit of cherry trees. The affected leaves may also become distorted or curl.',
        'treatment': 'To control Powdery mildew, it is recommended to prune out infected shoots and improve air circulation. Fungicides can also be used as a preventative measure.',
    },
    'Cherry healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Corn Cercospora leaf spot Gray leaf spot': {
        'symptoms': 'Cercospora leaf spot, also known as Gray leaf spot, causes elongated gray or tan lesions on the leaves of corn plants. As the disease progresses, the lesions may enlarge and coalesce.',
        'treatment': 'To manage Cercospora leaf spot, it is recommended to rotate crops, practice good sanitation, and use resistant corn varieties. Fungicides can also be applied if necessary.',
    },
    'Corn Common rust': {
        'symptoms': 'Common rust appears as small, circular to elongated orange pustules on the leaves, husks, and stems of corn plants. The pustules may rupture and release spores.',
        'treatment': 'Control measures for Common rust include planting resistant corn varieties, removing and destroying infected crop debris, and applying fungicides if necessary.',
    },
    'Corn Northern Leaf Blight': {
        'symptoms': 'Northern Leaf Blight causes cigar-shaped lesions with tan centers and dark brown borders on the leaves of corn plants. The lesions may expand and cover large portions of the leaves.',
        'treatment': 'To manage Northern Leaf Blight, it is important to practice crop rotation, use resistant corn hybrids, and apply fungicides if necessary. Removal of crop debris can also help reduce disease incidence.',
    },
    'Corn healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Grape Black rot': {
        'symptoms': 'Black rot causes black, circular lesions on the leaves, shoots, and fruit of grapevines. The lesions may have a distinct brown border and can lead to fruit rot and defoliation.',
        'treatment': 'Management of Black rot includes pruning out infected plant parts, improving air circulation, and applying fungicides according to the recommended schedule.',
    },
    'Grape Esca': {
        'symptoms': 'Esca, also known as Grapevine Leaf Stripe, causes yellow or red discoloration of the leaves. The affected leaves may also show dark streaks or necrotic areas. In advanced stages, the wood may decay.',
        'treatment': 'Controlling Esca is challenging, but practices such as pruning infected wood, using tolerant or resistant grape varieties, and applying specific protectant fungicides may help manage the disease.',
    },
    'Grape Leaf blight': {
        'symptoms': 'Leaf blight causes irregular brown or tan lesions on the leaves of grapevines. The lesions may have a water-soaked appearance and can expand to cover large areas of the leaf surface.',
        'treatment': 'To manage Leaf blight, it is recommended to prune out infected leaves, improve air circulation, and apply fungicides if necessary. Good canopy management practices can also help reduce disease incidence.',
    },
    'Grape healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Peach Bacterial spot': {
        'symptoms': 'Bacterial spot causes small, water-soaked lesions on the leaves, fruit, and twigs of peach trees. The lesions may turn brown or black and can be surrounded by a yellow halo. Infected fruit may develop raised, corky spots.',
        'treatment': 'Management of Bacterial spot involves pruning out infected twigs, improving air circulation, and applying copper-based bactericides during the dormant season and as directed during the growing season.',
    },
    'Peach healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Pepper bell Bacterial spot': {
        'symptoms': 'Bacterial spot causes dark, water-soaked lesions on the leaves, stems, and fruit of pepper plants. The lesions may enlarge and turn brown or black. Infected fruit may develop raised, corky spots.',
        'treatment': 'To control Bacterial spot, it is important to remove and destroy infected plant material, practice crop rotation, and apply copper-based bactericides as directed.',
    },
    'Pepper bell healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Potato Early blight': {
        'symptoms': 'Early blight causes brown, target-shaped lesions on the leaves of potato plants. The lesions may have concentric rings and can enlarge to cover large portions of the leaf surface.',
        'treatment': 'To manage Early blight, it is recommended to remove and destroy infected leaves, practice good crop rotation, and apply fungicides as necessary.',
    },
    'Potato Late blight': {
        'symptoms': 'Late blight causes dark, water-soaked lesions on the leaves, stems, and tubers of potato plants. The lesions may turn brown or black, and a white, fuzzy growth may develop under moist conditions.',
        'treatment': 'Management of Late blight includes removing and destroying infected plant material, applying fungicides preventatively, and providing good air circulation and drainage.',
    },
    'Potato healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Strawberry Leaf scorch': {
        'symptoms': 'Leaf scorch causes brown or purplish discoloration of the leaves, often starting from the margins and progressing inward. The affected leaves may also show necrotic spots or wilting.',
        'treatment': 'To manage Leaf scorch, it is important to remove and destroy infected leaves, practice good sanitation, and provide adequate irrigation and nutrition to the plants.',
    },
    'Strawberry healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    },
    'Tomato Bacterial spot': {
        'symptoms': 'Bacterial spot causes dark, water-soaked lesions on the leaves, stems, and fruit of tomato plants. The lesions may turn brown or black and can be surrounded by a yellow halo. Infected fruit may develop raised, corky spots.',
        'treatment': 'To control Bacterial spot, it is important to remove and destroy infected plant material, practice crop rotation, and apply copper-based bactericides as directed.',
    },
    'Tomato Early blight': {
        'symptoms': 'Early blight causes dark, concentric ring-shaped lesions on the leaves of tomato plants. The lesions may have a target-like appearance and can expand to cover large portions of the leaf surface.',
        'treatment': 'Management of Early blight involves removing and destroying infected leaves, providing good air circulation, and applying fungicides as necessary.',
    },
    'Tomato Late blight': {
        'symptoms': 'Late blight causes dark, water-soaked lesions on the leaves, stems, and fruit of tomato plants. The lesions may turn brown or black, and a white, fuzzy growth may develop under moist conditions.',
        'treatment': 'To manage Late blight, it is important to remove and destroy infected plant material, apply fungicides preventatively, and provide good air circulation and drainage.',
    },
    'Tomato Leaf Mold': {
        'symptoms': 'Leaf Mold causes yellowing of the upper leaf surface and the development of fuzzy, white to grayish-brown patches on the lower leaf surface. The affected leaves may also show curling or wilting.',
        'treatment': 'Management of Leaf Mold includes removing and destroying infected leaves, providing good air circulation, and avoiding overhead irrigation.',
    },
    'Tomato Septoria leaf spot': {
        'symptoms': 'Septoria leaf spot causes small, circular lesions with dark brown borders and gray centers on the leaves of tomato plants. The lesions may enlarge and coalesce, leading to defoliation.',
        'treatment': 'To control Septoria leaf spot, it is recommended to remove and destroy infected leaves, provide good air circulation, and apply fungicides as necessary.',
    },
    'Tomato Spider mites': {
        'symptoms': 'Spider mites cause stippling or yellowing of the leaves, often starting from the lower foliage. The affected leaves may also show webbing and eventually become bronzed or brown.',
        'treatment': 'To manage Spider mites, it is important to regularly monitor plants, provide adequate humidity, and use appropriate miticides as directed.',
    },
    'Tomato Target Spot': {
        'symptoms': 'Target Spot causes circular, dark brown lesions with concentric rings on the leaves, stems, and fruit of tomato plants. The lesions may enlarge and develop a shot-hole appearance.',
        'treatment': 'Management of Target Spot includes removing and destroying infected plant material, providing good air circulation, and applying fungicides as necessary.',
    },
    'Tomato Yellow Leaf Curl Virus': {
        'symptoms': 'Yellow Leaf Curl Virus causes yellowing and upward curling of the leaves. The affected plants may show stunted growth, and the fruit may be small and distorted.',
        'treatment': 'Controlling Yellow Leaf Curl Virus is challenging, but practices such as using resistant tomato varieties, controlling the vector insects, and removing infected plants can help reduce its spread.',
    },
    'Tomato mosaic virus': {
        'symptoms': 'Mosaic virus causes mottling, yellowing, and distortion of the leaves. The affected plants may show stunted growth, and the fruit may have irregular color patterns or be distorted.',
        'treatment': 'Controlling Tomato mosaic virus involves using virus-free seedlings, controlling vector insects, and removing infected plants to prevent its spread.',
    },
    'Tomato healthy': {
        'symptoms': 'No symptoms.',
        'treatment': 'No treatment required.',
    }
}


app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    # Set CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any origin
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'  # Allow GET and POST methods
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow the Content-Type header
    return response

@app.route("/", methods=['POST'])
def classify_image():
    img = request.files['img']
    img_path = save_temp_file(img)

    img = image.load_img(img_path, target_size=(150, 150))  # Resize the image to match VGG16's input shape
    img = image.img_to_array(img)
    img = preprocess_input(img)  # Preprocess the image according to VGG16's requirements

    img = np.expand_dims(img, axis=0)  # Add an extra dimension as the batch dimension

    predictions = leaf_disease_model.predict(img)

    label_index = np.argmax(predictions)
    label_name = label_names[label_index]
    accuracy = predictions[0][label_index] * 100

    # Get symptoms and treatment information for the predicted disease
    disease_symptoms = disease_info.get(label_name, {}).get('symptoms', 'Symptoms not available')
    disease_treatment = disease_info.get(label_name, {}).get('treatment', 'Treatment information not available')

    # Generate a bar graph of the prediction probabilities
    y_pos = np.arange(len(label_names))
    plt.bar(y_pos, predictions[0])
    plt.xticks(y_pos, label_names, rotation='vertical')
    plt.ylabel('Probability')
    plt.xlabel('Label')
    plt.title('Disease Classification')

    # Save the graph to a BytesIO object
    graph_bytes = io.BytesIO()
    plt.savefig(graph_bytes, format='png')
    plt.close()  # Close the figure

    # Encode the graph image as base64
    graph_base64 = base64.b64encode(graph_bytes.getvalue()).decode('utf-8')

    return jsonify({
        "Label Name": label_name,
        "Accuracy": accuracy,
        "Symptoms": disease_symptoms,
        "Treatment": disease_treatment,
        "Graph": graph_base64
    })


@app.route("/graph")
def get_graph():
    graph_path = save_temp_graph()
    return send_file(graph_path, mimetype='image/png')


def save_temp_file(file):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    temp_path = os.path.join(temp_dir, 'temp_file')
    dest_path = os.path.join(temp_dir, 'temp_file.bak')

    with open(temp_path, 'wb') as f:
        file.save(f)  # Save the FileStorage object to the temporary file

    with open(temp_path, 'rb') as source_file, open(dest_path, 'wb') as dest_file:
        shutil.copyfileobj(source_file, dest_file)  # Copy the content of the source file to the destination file

    try:
        os.unlink(temp_path)  # Remove the original file
    except PermissionError:
        pass  # Ignore the permission error and continue

    return dest_path


def save_temp_graph():
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    graph_path = os.path.join(temp_dir, 'graph.png')

    # Generate a bar graph of the prediction probabilities
    y_pos = np.arange(len(label_names))
    plt.bar(y_pos, [0] * len(label_names))  # Create an empty bar graph
    plt.xticks(y_pos, label_names, rotation='vertical')
    plt.ylabel('Probability')
    plt.xlabel('Label')
    plt.title('Disease Classification')

    # Save the graph to a file
    plt.savefig(graph_path)
    plt.close()  # Close the figure

    return graph_path


if __name__ == "__main__":
    app.run(debug=True)
