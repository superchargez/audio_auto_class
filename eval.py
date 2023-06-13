import keras
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

def extract_features(file_name, n_mels=128, n_fft=2048, hop_length=512, max_frames=10):
    audio, sample_rate = librosa.load(file_name, sr=8000)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')
    mels = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # Pad the melspectrogram if it has fewer frames than max_frames
    if mels_db.shape[1] < max_frames:
        mels_db = np.pad(mels_db, ((0, 0), (0, max_frames - mels_db.shape[1])), 'constant')

    # Truncate the melspectrogram if it has more frames than max_frames
    if mels_db.shape[1] > max_frames:
        mels_db = mels_db[:, :max_frames]

    return mels_db

test_data = []
test_labels = []

test_path = 'test/'
test_files = os.listdir(test_path)

for file in test_files:
    class_label, _, _ = file.split('_')
    mels_db = extract_features(os.path.join(test_path, file))
    test_data.append(mels_db)
    test_labels.append(int(class_label))

test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Add the channel dimension
test_data = test_data[..., np.newaxis]
model = keras.models.load_model(r'C:\Users\PTCL\projects\audio\data\best_model.h5')
y_pred = np.argmax(model.predict(test_data), axis=1)

f1 = f1_score(test_labels, y_pred, average='weighted')
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='weighted')
recall = recall_score(test_labels, y_pred, average='weighted')

print("Test F1 Score: ", f1)
print("Test Accuracy: ", accuracy)
print("Test Precision: ", precision)
print("Test Recall: ", recall)
