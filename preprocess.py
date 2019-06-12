import numpy as np
import librosa
import os

SR = 22050
time = 5 # seconds

As = [] # 0
for f in os.listdir('./data/train/A'):
    audio, _ = librosa.load('./data/train/A/' + f)
    print(f)
    for i in range(audio.shape[0] // (SR * time)):
        As.append(audio[i * SR * time : (i + 1) * SR * time])

Bs = [] # 1
for f in os.listdir('./data/train/B'):
    audio, _ = librosa.load('./data/train/B/' + f)
    print(f)
    for i in range(audio.shape[0] // (SR * time)):
        Bs.append(audio[i * SR * time : (i + 1) * SR * time])

print("1")

Aspectrograms = []
for i in range(len(As)):
    audio = As[i]
    spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, hop_length=256, n_fft=512, n_mels=64)**2)
    Aspectrograms.append(spectrogram)

print("2")

Bspectrograms = []
for i in range(len(As)):
    audio = Bs[i]
    spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, hop_length=256, n_fft=512, n_mels=64)**2)
    Bspectrograms.append(spectrogram)

print("3")

specs = np.array(Aspectrograms + Bspectrograms)
specs = specs.reshape((specs.shape[0], specs.shape[1] * specs.shape[2]))
labels = np.array([0] * len(As) + [1] * len(Bs))

data = np.concatenate((specs, labels.reshape((labels.shape[0], 1))), axis=0)
np.random.shuffle(data)

np.savez('./data/data.npz', data=data)
