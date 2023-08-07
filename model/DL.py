# %%
import numpy as np
import os
from tqdm import tqdm

# %% [markdown]
# ## Loading Descriptions
# Image Caption are stored in Flickr8k.token.txt seperated by a new line
# eachline consist of name of the image followed by space(tab) and description

# %%
# load_description_data function
# Input: description_path (str) - Path to the description file
# Output: mapping (dict) - Mapping of Image and its descriptions (one image can have multiple descriptions)
def load_description_data(description_path: str):
    raw_text = open(description_path, 'r', encoding='utf-8').read()
    
    # Mapping of Image and its descriptions (one image can have multiple descriptions)
    mapping = dict()

    # Iterate through each line of the file
    for line in raw_text.split('\n'):
        token = line.split("\t") # Seperate Image name and Description
        
        # Filter out lines with less than 2 elements (i.e. no image name or description)
        if len(line) < 2:
            continue

        image_id          = token[0].split('.')[0]
        image_description = token[1]

        # If image id is not present in the mapping, add it
        if image_id not in mapping:
            mapping[image_id] = list()
        
        # Add description to the mapping
        mapping[image_id].append(image_description)
    
    return mapping

# %% [markdown]
# ## Cleaning the text
# 
# Remove noise so that our NLP machine can easily detect pattern in the text. In the case of Flicker8K dataset noise will come in numbers and special character which will be quite are for machine to understand (I am not OpenAI lmao)

# %%
# clean_descriptions_data function
# Input: description_mapping (dict) - Mapping of Image and its descriptions from load_description_data function
# Output: None - Modifies the description_mapping in place
def clean_descriptions_data(description_mapping: dict):
    # image_id (str) - Image name
    # descriptions (list) - List of descriptions for the image
    for image_id, descriptions in description_mapping.items():
        # Iterate through each description
        for i in range(len(descriptions)):
            description = descriptions[i]                                                          # Query description
            description = description.split()                                                      # Split description into words
            description = [word.lower() for word in description if len(word)>1 and word.isalpha()] # Remove single character words and punctuations (i.e. 'a', '!')
            description = ' '.join(description)                                                    # Join words back into a sentence
            descriptions[i] = description                                                          # Replace description with cleaned description

# %% [markdown]
# ## Generate Unique Vocabs
# Vocabulary is a set of unique word present in the text corpus (all vocab our model will know - Similar to Human you don't understand Chinese Words lmao)

# %%
def get_vocabs(description_mapping: dict):
    # Create a set of all unique words
    vocab = set()          

    # Iterate through each image and its descriptions
    for image_id in description_mapping.keys():
        # Iterate through each description
        for description in description_mapping[image_id]:
            # Union all words in the description to the set (Word can be portrayed as vocabulary - I am not a linguist pls don't take my word for it 💀)       
            vocab.update(description.split())
    return vocab

# %% [markdown]
# ## Load text data

# %%
# Load Raw Description Data
descriptions = load_description_data('Flickr_Data/Flickr_TextData/Flickr8k.token.txt')
print("Raw Description Data Demo    :", descriptions['1000268201_693b08cb0e'])

# Clean Description Data
clean_descriptions_data(descriptions)
print("Cleaned Description Data Demo:", descriptions['1000268201_693b08cb0e'])

# Get Vocabularies
vocab = get_vocabs(descriptions)
print("Vocabularies Demo            :", list(vocab)[:10])

# %% [markdown]
# ## Load Dataset Descriptions
# Map images in the training set to their corresponding descriptions
# Create a list of training_images file name and then create a empty dictionary and map to their descriptions using image name as key

# %%
import glob

# Create a list of all image names
images_dir = 'Flickr_Data/Images'
image_paths = glob.glob(images_dir + '/*.jpg')

# List of all training image names
train_image_texts = 'Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
train_image_names = open(train_image_texts, 'r').read().split('\n')

# List to store paths of training images
train_image_paths = []
for image_path in image_paths:
    # Check if image is in the training set by comparing its name
    if image_path[len(images_dir) + 1:] in train_image_names:
        train_image_paths.append(image_path)

# Function to load descriptions for the training dataset
# Input: description_mapping (dict) - Mapping of image names to descriptions
#        image_paths (list) - List of image paths
# Output: dataset (dict) - Mapping of image names to preprocessed descriptions
def load_description_dataset(description_mapping: dict, image_paths: list):
    dataset = dict()

    for image_id, descs in description_mapping.items():
        
        expected_path = image_id + '.jpg'

        # Check if image in list of image of interest to built dataset
        if expected_path in image_paths:

            # If image id is not present in the dataset, add it
            if image_id not in dataset:
                dataset[image_id] = list()
            
            # Add description to the dataset
            for desc in descs:
                dataset[image_id].append('<start> ' + desc + ' <end>')
            
    return dataset
            
# Load Training Dataset
# Sub Sample of variable descriptions for those who still don't understand 🤣
train_descriptions = load_description_dataset(descriptions, train_image_names)

# %% [markdown]
# ## Image Features Extraction
# For our model to understand the image we have to extract feature from the image in a machine understandable way not human aka computer do not have eyes lmao.
# We will use pretrained InceptionV3 to extract these feature by taking output of the model that would be feed to classification model as the features

# %%
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model

# Load InceptionV3 Model pretrained on ImageNet dataset
base_model = InceptionV3(weights='imagenet')

# Remove the last 2 layers of the model (i.e. the classification layers -> softmax and predictions)
model_inception = Model(base_model.input, base_model.layers[-2].output)


# Load Image for InceptionV3
def preprocess_img(img_path: str):
    
    # Inception V3 expects image to be of size (1, 299, 299, 3)
    img = load_img(img_path, target_size=(299, 299))

    x = img_to_array(img)           # Convert image to numpy array    
    x = np.expand_dims(x, axis=0)   # Add one more dimension to the image (i.e. (299, 299, 3) -> (1, 299, 299, 3))

    x = preprocess_input(x)         # Preprocess the image for InceptionV3

    # Image ready for InceptionV3
    return x

# Perform Feature Extraction using InceptionV3 aka encode image with DL model
def encode_image(img_path: str):

    img = preprocess_img(img_path)          # Preprocess image for InceptionV3

    code = model_inception.predict(img, verbose=0)    # Encode image with InceptionV3
    code = np.reshape(code, code.shape[1])  # Reshape from (1, 2048) to (2048, )

    return code


# %%
# Encode all training images
train_image_encoded = {}
for img in tqdm(train_image_paths):
    train_image_encoded[img[len(images_dir) + 1:].replace('.jpg', '')] = encode_image(img)


# %% [markdown]
# ## Tokenizing the Vocabulary
# Tokenized all the vocab presented in our Corpus

# %%
# Create a list of all training captions [caption, caption, caption, ...]
all_train_captions = [caption for captions in train_descriptions.values() for caption in captions]

# Filter out words that occur less than 10 times to reduce vocabulary size
threshold = 10

# Create a dictionary of word and its count
word_counts = {}
for caption in all_train_captions:
    for word in caption.split():
        word_counts[word] = word_counts.get(word, 0) + 1

# Filtering
vocab = [word for word in word_counts if word_counts[word] >= threshold]

word2int = {word: index for index, word in enumerate(vocab, 1)}
int2word = {index: word for index, word in enumerate(vocab, 1)}

# Find max length of a caption in the training set
max_length = max(len(caption.split()) for captions in train_descriptions.values() for caption in captions)

# %% [markdown]
# ## X-Y Extraction

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Prepare dataset for training
# X1 - Image Features
# X2 - Input Sequence
# y  - Output Word
X1, X2, y = [], [], []

# Iterates over each train_descriptions retrieves the corresponding iamge feature from train_image_encoded dictionary
for key, captions in train_descriptions.items():
    # Extract image features vector
    img = train_image_encoded[key]

    # Iterate through each description for the image
    for caption in captions:
        seq = [word2int[word] for word in caption.split() if word in word2int]  # Convert caption to sequence of integers
        
        # Iterate through each word in the sequence to generate what come next for each words
        for i in range(1, len(seq)):

            in_seq, out_seq = seq[:i], seq[i]  # Split sequence into input and output pair

            # FAQ: Why one-hot encode the output sequence?
            # Ans: LSTM need one hot output while embedding layer need integer input hence we need to one-hot encode the output sequence but not the input sequence
            # Pad input sequence to make it the same length as max_length
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # One-hot encode output sequence
            out_seq = to_categorical([out_seq], num_classes=len(vocab) + 1)[0]

            # Append to dataset
            X1.append(img)
            X2.append(in_seq)
            y.append(out_seq)


# Convert to numpy array
X1, X2, y = np.array(X1), np.array(X2), np.array(y)


# %% [markdown]
# ## GloVe Vector Embeddings
# 
# GloVe stands for global vectors for word representation helps computers understand the meaning of words by representing them as numbers.
# Vector representation of Corgi and Husky will be quite similar
# 

# %%
glove_path = 'glove.6B.200d.txt'
glove = open(glove_path, 'r').read().split('\n')

# Create a dictionary of word and its embedding
embeeding_index = {}

# Doing something similar to opening a dictionary for word meaning - {word} {vector}
for line in tqdm(glove):
    try:
        values = line.split()
        # Get both the word and the vector
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')

        # Add to dictionary
        embeeding_index[word] = vector
    except IndexError:
        pass
# Generate embedding matrix
embedding_dim = 200
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
for word, index in word2int.items():
    embedding_vector = embeeding_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# %% [markdown]
# ## DEEP DARK LEARNING MODEL
# Our model will have 3 major steps
# 1. Processing Sequence of text
# 2. Extracting the feature vector from the image
# 3. Decoding the output by concatenating 1. and 2.
# 
# Model reference from article by dark_coder88

# %%
# Print Dataset Dimension
print('X1.shape:', X1.shape)    # Image Features
print('X2.shape:', X2.shape)    # Input Sequence
print('y.shape:', y.shape)      # Output Word

# %%
from tensorflow.keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation, concatenate, BatchNormalization, Input, Concatenate


# Why len(vocab) + 1?
# Ans: Because we need to add a padding token to the vocabulary

# 1. Process Image
input_1     = Input(shape=(2048,))  # Image Features
dropout_1   = Dropout(0.2)(input_1) # Dropout Layer
dense_1     = Dense(256, activation='relu')(dropout_1) # Fully Connected Layer

# 2. Process Text 
input_2     = Input(shape=(max_length,)) # Input Sequence
embedding_2 = Embedding(len(vocab)+1, embedding_dim, mask_zero=True)(input_2) # Embedding Layer
dropout_2   = Dropout(0.2)(embedding_2) # Dropout Layer
lstm_2      = LSTM(256)(dropout_2) # LSTM Layer

# 3. Merge Image and Text 
merge_3       = Concatenate()([dense_1, lstm_2]) # Concatenate Layer
dense_3       = Dense(256, activation='relu')(merge_3) # Fully Connected Layer
outputs       = Dense(len(vocab)+1, activation='softmax')(dense_3) # Output Layer

# Model
model = Model(inputs=[input_1, input_2], outputs=outputs)

# Set embedding layer weights
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')


# %%
import json

def save_dictionary_as_json(dictionary, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(dictionary, file)
        print("Dictionary saved as JSON successfully.")
    except Exception as e:
        print(f"Error saving the dictionary as JSON: {str(e)}")

# Example usage
save_dictionary_as_json(word2int, 'word2int.json')
save_dictionary_as_json(int2word, 'int2word.json')


# %% [markdown]
# ### Train Model

# %%
from tensorflow.keras.callbacks import TensorBoard

log_dir = "./logs"  # Directory where the log files will be saved
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')

# Start training the model with the TensorBoard callback
model.fit([X1, X2], y, epochs=50, batch_size=256, verbose=2, callbacks=[tensorboard_callback])

# %%
model.save('models')

# %% [markdown]
# ## Predicting

# %%
model.load_weights('saved')

# %%
def predict(img_file):

    # Load and resize image
    img = encode_image(img_file).reshape((1, 2048))

    start = '<start>'

    for i in range(max_length):
        
        # Convert caption to sequence of integers
        seq = [word2int[word] for word in start.split() if word in word2int]
        
        # Pad sequence to make it the same length as max_length to feed into the model
        seq = pad_sequences([seq], maxlen=max_length)

        predict_word = model.predict([img, seq], verbose=0)
        predict_word = np.argmax(predict_word)
        word = int2word[predict_word]

        start += ' ' + word

        # Model predicts the word <end> we break the loop
        if word == '<end>':
            break
    
    final = start.split()
    
    # Remove <start> and <end> from the final caption
    final = final[1:-1]

    return ' '.join(final)


# %%
# Test Image from the test set

# Randomly select an image from the test set
images_dir = 'Flickr_Data/Images'
image_paths = glob.glob(images_dir + '/*.jpg')

# Randomly select an image that is not in the training set
test_paths = list(set(image_paths) - set(train_image_paths))

# Randomly select an image
img_path = test_paths[np.random.randint(0, len(test_paths))]

# Display the image
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(img_path)

# Print the image caption
print('Caption:', predict(img_path))

plt.imshow(img)


# %%
predict('g.jpg')

# %%



