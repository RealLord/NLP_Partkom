import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalMaxPool1D, Conv1D
#from keras.layers.embeddings import Embedding
from keras.layers import Embedding
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences
from keras import utils

from tensorflow.python.client import device_lib


def pir(s):
    return pd.DataFrame({'a':s.value_counts(), 'per':s.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'})


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.function(jit_compile=True)  # Включение XLA
except:
    # Invalid device or cannot modify virtual devices once initialized.
    exit(100)

##################
debug = True
##################

df_draft = pd.read_csv('data/goods_01.zip', sep=';', header=0, encoding='utf-8', dtype={"id": int, "group_code": int, "desc": "string"}, compression="zip")

#print(df_draft.head(10))
#print(df_draft.group_code.value_counts())
#print('Max lenght desc: %s' % df_draft["desc"].str.len().max())

sentences = df_draft['desc']
y = df_draft['group_code']

if debug:
    print(y.value_counts())

sentences_train, sentences_test, train_y, test_y = train_test_split(sentences, y, test_size=0.30, stratify=y)

if debug:
    print("Train_y codes:" )
    print(pir(train_y))
    print("Test_y codes:" )
    print(pir(test_y))

tokenize = Tokenizer(num_words=5000)
texts = pd.concat([sentences_train, sentences_test], axis=0).astype("str")
tokenize.fit_on_texts(texts)

X_train = tokenize.texts_to_sequences(pd.concat([sentences_train], axis=0).astype("str"))
X_test = tokenize.texts_to_sequences(pd.concat([sentences_test], axis=0).astype("str"))
vocab_size = len(tokenize.word_index) + 1

maxlen = 128
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# вариант с энкодингом
encoder = LabelEncoder()
#encoder.fit(train_y)

y_train = encoder.fit_transform(train_y.values)
y_test = encoder.transform(test_y.values)

# y_train = encoder.transform(train_y)
# y_test = encoder.transform(test_y)
# num_classes = np.max(y_train) + 1
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test  = tf.keras.utils.to_categorical(y_test,  num_classes)


embedding_dim = 30
model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model4.add(GlobalMaxPool1D())
model4.add(Dropout(0.2))
model4.add(Dense(32, activation='relu'))  #16 is default
model4.add(Dropout(0.2))
model4.add(Dense(431, activation='softmax'))  # 431 - number of unique group_code

model4.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model4.summary()
with tf.device('/gpu:0'):
    history_4 = model4.fit(X_train, y_train,
                        batch_size=32,  # 32 - default
                        epochs=1,  # 15 - default
                        validation_data=(X_test, y_test))

model4.save('model4.save')  # Saving model!
if debug:
     print('Go to evaluate')

with tf.device("/cpu:0"):
    loss, accuracy = model4.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model4.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


