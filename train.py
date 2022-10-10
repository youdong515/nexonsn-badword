#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import json
import joblib
import inspect
import seaborn as sns
import pandas as pd 
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Flatten, Conv1D, Dropout, Embedding, Input, Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K

data = pd.read_csv("data_processed.csv")
label = data['badword']

max_length = 64
label_num = 1
learning_rate = 0.001
batch_size = 32
epochs = 10

token = Tokenizer(num_words=None, filters=r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\u4e00-\u9fff]|(.)\1{2,}', char_level=True)
token.fit_on_texts(data['txt'])
word_index = token.word_index

seq = token.texts_to_sequences(data['txt'].astype(str))
paded = pad_sequences(seq, maxlen=max_length, padding='post')

x_train, x_valid, y_train, y_valid = train_test_split(paded, label, test_size=0.2, random_state=1234)

# model
inputs = Input(shape=(max_length, ))
embeded = Embedding(len(token.word_index)+1, 32, input_length=max_length)(inputs)

conv = Conv1D(filters=32, kernel_size=32, activation='relu', padding='same')(embeded)
conv = Dropout(0.5)(conv)
conv = Conv1D(filters=48, kernel_size=16, activation='relu', padding='same')(conv)
conv = Dropout(0.5)(conv)
conv = Conv1D(filters=64, kernel_size=6, activation='relu', padding='same')(conv)
conv = Dropout(0.5)(conv)
conv = Conv1D(filters=128, kernel_size=2, activation='relu', padding='same')(conv)
conv = Dropout(0.5)(conv)

flat = Flatten()(conv)
dense = Dense(128, activation='relu')(flat)
dense = Dense(label_num, activation='sigmoid')(dense)

model = Model(inputs, dense)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

y_valid = pd.DataFrame(y_valid).reset_index(drop=True)

# test
predict_val = model.predict(x_valid)
predict_val = pd.DataFrame(predict_val)
predict_val.columns = ['value']
rd = pd.Series(np.floor(predict_val['value']*10)/10, name='rd')
predict_val['round_value'] = round(predict_val['value'], 0)
predict_val = pd.concat([y_valid, predict_val, rd], axis=1, ignore_index=False)

predict_val.columns = ['label', 'value', 'round_value', 'round_score']

# f1 score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

y_true = predict_val['label'].tolist()
y_pred = predict_val['round_value'].tolist()

confusion_matrix = confusion_matrix(y_true, y_pred, labels=[1,0])

ac = accuracy_score(y_true, y_pred)
rs = recall_score(y_true, y_pred)
ps = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": ac, "recall": rs, "precision":ps, "f1_score":f1}, outfile)

# Bar plot by region

cm_matrix = pd.DataFrame(data=confusion_matrix, columns=['Predict Badword:1', 'Predict Badword:0'], 
                                 index=['Actual Badword:1', 'Actual Badword:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.savefig("by_region.png",dpi=80)
