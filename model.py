MAX_LENGTH = 35

import numpy as np
import json
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation, concatenate, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model

import base64
from PIL import Image
from io import BytesIO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class CaptionGenerator:
    def __init__(self, model_dir: str, data_dir: str):
        # Initialize word-to-index and index-to-word mappings
        self.initialize_data(data_dir)
        # Initialize InceptionV3 model for image feature extraction
        self.initialize_inception()
        # Initialize caption generation model
        self.initialize_model(model_dir)
        

    def predict(self, img_b64: str):
        img = self.decode_base64_image(img_b64)

        # Load and preprocess image
        img = self.encode_image(img).reshape((1, 2048))

        start = '<start>'
        for i in range(MAX_LENGTH):
            # Convert the current caption to a sequence of integers
            seq = [self.word2int[word] for word in start.split() if word in self.word2int]
            # Pad the sequence to match the required input length of the model
            seq = pad_sequences([seq], maxlen=MAX_LENGTH)

            # Generate the next word prediction
            predict_word = self.model.predict([img, seq], verbose=0)
            predict_word = np.argmax(predict_word)
            word = self.int2word[str(predict_word)]

            # Append the predicted word to the current caption
            start += ' ' + word

            # If the model predicts the end token, stop generating further words
            if word == '<end>':
                break

        # Remove the start and end tokens from the final caption
        final = start.split()[1:-1]
        return ' '.join(final)

    def initialize_inception(self):
        # Load InceptionV3 model pre-trained on ImageNet
        base_model = InceptionV3(weights='imagenet')
        # Remove the last 2 layers of the model (classification layers)
        self.model_inception = Model(base_model.input, base_model.layers[-2].output)

    def initialize_model(self, model_dir: str):
        # Define the caption generation model architecture
        input_1 = Input(shape=(2048,))  # Image Features
        dropout_1 = Dropout(0.2)(input_1)  # Dropout Layer
        dense_1 = Dense(256, activation='relu')(dropout_1)  # Fully Connected Layer

        input_2 = Input(shape=(MAX_LENGTH,))  # Input Sequence
        embedding_2 = Embedding(len(self.vocab) + 1, 200, mask_zero=True)(input_2)  # Embedding Layer
        dropout_2 = Dropout(0.2)(embedding_2)  # Dropout Layer
        lstm_2 = LSTM(256)(dropout_2)  # LSTM Layer

        merge_3 = Concatenate()([dense_1, lstm_2])  # Concatenate Layer
        dense_3 = Dense(256, activation='relu')(merge_3)  # Fully Connected Layer
        outputs = Dense(len(self.vocab) + 1, activation='softmax')(dense_3)  # Output Layer

        # Create the caption generation model
        self.model = Model(inputs=[input_1, input_2], outputs=outputs)

        # Set the embedding layer weights
        self.model.layers[2].set_weights([self.embedding_matrix])
        self.model.layers[2].trainable = False

        # Compile the model with appropriate loss and optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        # Load pre-trained weights for the model
        self.model.load_weights(model_dir)

    def initialize_data(self, data_dir: str):
        # Load word-to-index mapping
        with open(data_dir + 'word2int.json', 'r') as f:
            self.word2int = json.load(f)
        
        # Load index-to-word mapping
        with open(data_dir + 'int2word.json', 'r') as f:
            self.int2word = json.load(f)

        # Load Embedding Matrix
        self.embedding_matrix = np.load(data_dir + 'embedding_matrix.npy')

        # Read Vocabulary
        with open(data_dir + 'vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)


    def encode_image(self, img: Image):
        # Load and preprocess the image using InceptionV3
        img = np.expand_dims(img_to_array(img), axis=0)
        img = preprocess_input(img)
        # Extract image features using InceptionV3
        code = self.model_inception.predict(img)
        code = np.reshape(code, code.shape[1])
        return code
    
    def decode_base64_image(self, img_base64: str):
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data)).resize((299, 299))
        return img
    
