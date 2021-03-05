import pickle
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import torch

from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# Class for data preparations.
class Data:
    def __init__(self, movies):
        self.movies = movies

    def get_place(self):
        place = self.movies.get('place')
        place = pd.DataFrame(place).astype('float')

        return place

    def get_cast(self):
        cast = self.movies.get('cast')
        cast = pd.DataFrame(cast).astype('float')

        return cast

    def get_action(self):
        action = self.movies.get('action')
        action = pd.DataFrame(action).astype('float')

        return action

    def get_audio(self):
        audio = self.movies.get('audio')
        audio = pd.DataFrame(audio).astype('float')

        return audio

    def get_scene_transition(self):
        scene_ = self.movies.get('scene_transition_boundary_prediction')
        scene_ = pd.DataFrame(scene_).astype('float')
        scene_.loc[-1] = 0.0000
        scene_.index = scene_.index + 1
        scene_ = scene_.sort_index()

        return scene_

    def get_ground_truth(self):
        scene_ = self.movies.get('scene_transition_boundary_ground_truth')
        scene_ = pd.DataFrame(scene_).astype('str')
        scene_.loc[-1] = ['False']
        scene_.index = scene_.index + 1
        scene_ = scene_.sort_index()

        return scene_

    def separate_results(self, y, index):
        length = len(self.movies.get('scene_transition_boundary_prediction')) + 1
        scene_transition_boundary_prediction = y[index: index + length]
        scene_transition_boundary_prediction = np.delete(scene_transition_boundary_prediction, 0)
        scene_transition_boundary_prediction = torch.from_numpy(scene_transition_boundary_prediction)
        self.movies['scene_transition_boundary_prediction'] = scene_transition_boundary_prediction
        index = length + index

        return index


path = os.getcwd()
data_path = path + '/data'
data_files = os.listdir(data_path)
movie_data = []

for file in data_files:
    file = data_path + '/' + file
    with open(file, 'rb') as f:
        data = pickle.load(f)
        movie_data.append(Data(data))

# Reads data if file is already exists.
if os.path.isfile('PlaceData'):
    data = open('PlaceData', 'rb')
    place_data = pickle.load(data)
    data = open('ActionData', 'rb')
    action_data = pickle.load(data)
    data = open('CastData', 'rb')
    cast_data = pickle.load(data)
    data = open('AudioData', 'rb')
    audio_data = pickle.load(data)
    data.close()

# Prepares the data. 
else:
    place_data = []
    cast_data = []
    action_data = []
    audio_data = []
    for movie in movie_data:
        place_data.append(movie.get_place())
        cast_data.append(movie.get_cast())
        action_data.append(movie.get_action())
        audio_data.append(movie.get_audio())

    place_data = pd.concat(place_data)
    cast_data = pd.concat(cast_data)
    action_data = pd.concat(action_data)
    audio_data = pd.concat(audio_data)

    data = open('PlaceData', 'ab')
    pickle.dump(place_data, data)
    data = open('CastData', 'ab')
    pickle.dump(cast_data, data)
    data = open('ActionData', 'ab')
    pickle.dump(action_data, data)
    data = open('AudioData', 'ab')
    pickle.dump(audio_data, data)
    data.close()

# Reads the ground truth and already predicted probabilities.
scene_transition = []
ground_truth = []
for movie in movie_data:
    scene_transition.append(movie.get_scene_transition())
    ground_truth.append(movie.get_ground_truth())

scene_transition_data = pd.concat(scene_transition)
ground_truth_data = pd.concat(ground_truth)
scene_transition_data = scene_transition_data.to_numpy()
print('Reading the data.')

# Feature reduction is done by PCA.
pca = PCA(n_components=200)
place_data = pca.fit_transform(place_data)
cast_data = pca.fit_transform(cast_data)
action_data = pca.fit_transform(action_data)
audio_data = pca.fit_transform(audio_data)
print('Feature Reduction is done.')

dataset = (place_data, cast_data, action_data, audio_data, scene_transition_data)
dataset_ = np.concatenate(dataset, axis=1)
print('Dataset is Collected.')

# Encoding the y value for training.
encoder = LabelEncoder()
ground_truth_data = encoder.fit_transform(ground_truth_data)

# Model for training. 
model = tf.keras.models.Sequential()
model.add(Dense(input_dim=801, units=1024, activation='tanh'))
model.add(Dense(512, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model for 10 Epochs with batch size of 128.
print('Training the Model.')
model.fit(dataset_, ground_truth_data, epochs=10, batch_size=128)

# Predicting the results.
y_pred = model.predict(dataset_)
print('Predicting the results.')

# Storing the results in results folder. 
index = 0
for movie in movie_data:
    index = movie.separate_results(y_pred, index)
    index = index
    movies = movie.movies
    file = open('results/' + str(movies.get('imdb_id')) + '.pkl', 'wb')
    pickle.dump(movies, file)
    file.close()

print('Model is Trained and Program is completed. Now run evaluate.py to see the model performance.')