import re
import time
import json
import joblib
import inspect
import seaborn as sns
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, Flatten, Conv1D, Dropout, Embedding, Input, Activation, add, multiply, 
                                     BatchNormalization, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPool1D,
                                     GRU, Bidirectional)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

data = pd.read_csv("./base_english.csv")

def get_bi_char_words(input_txt, custom_period='%', pad_char_reg_exp='[a-zA-Z0-9]'):
    """
    get_bi_char_words('abcd다0')
    'abcd%다0%'
    """
    def is_pad_character(char):

        if not isinstance(char, str):
            raise ValueError('str type char should be given.')
        elif len(char) != 1:
            raise ValueError('char should be given.')

        else:
            pass

        if re.search(pad_char_reg_exp, char):
            return True
        else:
            return False

    tmp_output_txt = []
    for idx, current_char in enumerate(input_txt):
        if idx == len(input_txt) - 1:
            next_char = None
        else:
            next_char = input_txt[idx + 1]

        if is_pad_character(current_char):  # 현재 문자가 2글자 토큰 대상이고
            if next_char is None or not is_pad_character(next_char):  # 현재 문자가 마지막 문자이거나, 다음문자가 대상이 아니면
                tmp_output_txt.append(current_char + custom_period)  # 현재 문자에 pad_char 붙여서 넣는다.
            else:
                tmp_output_txt.append(current_char + next_char)  # 현재문자와 다음 문자 모두 대상이므로 붙여서 넣는다.
        else:  # 한글 문자 같은 경우 문자 하나만 넣는다.
            tmp_output_txt.append(current_char)

        if next_char is not None:
            tmp_output_txt.append(' ')  # 단어 구분자를 넣는다. 마지막이 아닌때만 넣는다.

    return ''.join(tmp_output_txt)

def stride_w_custom_period(
        string,
        max_len=64,
        filters=r'[\?\.\,\<\>\\\|\=\+\-\_\(\)\{\}\[\]\&\^\#\!\`\~\'\"\:\;\/\%]',
        custom_period='%',
        timing=False):
    """
    stride_w_custom_period('[Abc0다abc')
    returns 'ab bc c0 0% %다'
    """

    if len(custom_period) != 1:
        raise ValueError('custom_period must be a letter, not a word.')
    if custom_period not in filters:
        raise ValueError('custom_period must be in the filters.')

    string = string[:max_len]  # maxlen 만큼만 사용
    string = re.sub(filters, r'', string)  # filter에 대당하는 문자는 ''으로 대체
    string = re.sub(r'(.)\1{2,}', r'\1\1', string)  # ???
    string = re.sub(r' ', '', string)  # 빈칸을 없앤다.
    string = get_bi_char_words(string, custom_period, pad_char_reg_exp='[a-zA-Z0-9]')

    return string

max_length = 64
label_num = 1
learning_rate = 0.001
batch_size = 32
epochs = 10

txt = data['txt']
label = data['badword']
txt_len = len(txt)
new_frame = []

for cnt in range(txt_len):
        raw_string = txt[cnt]
        result = stride_w_custom_period(raw_string, timing=False)
        
        new_frame.extend([result])
 
data = pd.DataFrame(new_frame)
data.columns = ['txt']
data = pd.concat([label, data], axis=1)
data.columns = ['badword', 'txt']   

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
