import os
import librosa
import numpy as np
import keras

def extract_features(file_name, n_mels=128, n_fft=2048, hop_length=512, max_frames=100, segment_length=0.2):
    audio, sample_rate = librosa.load(file_name, sr=8000)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')
    
    # Split audio into segments
    segment_samples = int(segment_length * sample_rate)
    segments = [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]
    
    # Load the model
    model = keras.models.load_model(r'C:\Users\PTCL\projects\audio\data\best_model.h5')
    
    results = []
    for i, segment in enumerate(segments):
        mels = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mels_db = librosa.power_to_db(mels, ref=np.max)

        # Pad the melspectrogram if it has fewer frames than max_frames
        if mels_db.shape[1] < max_frames:
            mels_db = np.pad(mels_db, ((0, 0), (0, max_frames - mels_db.shape[1])), 'constant')

        # Truncate the melspectrogram if it has more frames than max_frames
        if mels_db.shape[1] > max_frames:
            mels_db = mels_db[:, :max_frames]
        
        # Run audio classifier on log mels to get class of segment
        segment_class = run_audio_classifier(mels_db, model)
        
        # Time at which class was determined
        time = i * segment_length
        
        results.append((segment_class, time, i))
    
    return results



from tensorflow import keras

def run_audio_classifier(mels_db, model):
    # Reshape the input to match the model's expected input shape
    mels_db = mels_db.reshape(1, mels_db.shape[0], mels_db.shape[1], 1)
    
    # Run the model on the input
    predictions = model.predict(mels_db)
    
    # Get the class with the highest probability
    segment_class = np.argmax(predictions)
    
    return segment_class


file = r"C:\Users\PTCL\projects\audio\data\test\7_george_29.wav"
results = extract_features(file_name=file, n_mels=128, n_fft=2048, hop_length=512, max_frames=10, segment_length=.55)

for result in results:
    print(f'class: {result[0]}, time: {result[1]}, segment #: {result[2]}')
