import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define function to get word from index
def get_word_from_index(index, tokenizer):
    return next(
        (word for word, idx in tokenizer.word_index.items() if idx == index), None
    )

# Define function to predict caption
def predict_caption(model, image_features, tokenizer, max_caption_length):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        caption += " " + predicted_word
        if predicted_word is None or predicted_word == "endseq":
            break
    return caption.replace("startseq", "").replace("endseq", "")

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', caption="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', caption="No selected file")

        if file:
            image = load_img(file, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            image_features = mobilenet_model.predict(image, verbose=0)

            # Max caption length
            max_caption_length = 34

            # Generate caption
            caption = predict_caption(model, image_features, tokenizer, max_caption_length)

    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
