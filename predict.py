import train_DNN as t
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
import scipy.io.wavfile as wav
from utilities import get_data, class_labels
import numpy as np
import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


data= "test.wav"
def read_wav(filename):
    return wav.read(filename)



mslen = 32000  # Empirically calculated for the given dataset
def get_data1(flatten=False, mfcc_len=39):


    fs, signal = read_wav(data)
    s_len = len(signal)
    if s_len < mslen:
        pad_len = mslen - s_len
        pad_rem = pad_len % 2
        pad_len /= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mslen
        pad_rem = pad_len % 2
        pad_len /= 2
        signal = signal[pad_len:pad_len + mslen]
    mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

    if flatten:
        # Flatten the datas
        mfcc = mfcc.flatten()
    return np.array(mfcc)




# Read data
global x_train, y_train, x_test, y_test
x_train, x_test, y_train, y_test = get_data(flatten=False)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_tesst)

model = t.get_model('LSTM', x_train[0].shape)
print(x_train[0].shape)
model.summary()
model.load_weights('best_model.h5')
class_labels = ["Neutral", "Angry", "Sad" , "Happy" ]
a=get_data1()
print(a)
a=np.expand_dims(a,axis=0)

prediction = model.predict(a)
prob=prediction[0]
print(prob)
maximum = max(prob)
for i in range(len(prob)):
    if(prob[i] == maximum):
        p=i
print("Oh You Are Verrrry ",class_labels[p])









