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
#print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.function(jit_compile=True)  # Включение XLA
except:
    # Invalid device or cannot modify virtual devices once initialized.
    exit(100)

####################
debug = True
####################

# df_group = pd.read_csv('data/goods_code_groups.csv', sep=';', header=0, encoding='utf-8', dtype={"id": int, "group_code": int})
# df_group.drop('id', axis=1, inplace=True)

embedding_dim = 30
model4 = tf.keras.models.load_model('model4.save')
model4.summary()

input_text = "696 0221118307 катушка зажигания"  # 654 код группы

tokenize = Tokenizer(num_words=5000)
tokenize.fit_on_texts(input_text)

X_train = tokenize.texts_to_sequences(input_text)
vocab_size = len(tokenize.word_index) + 1

maxlen = 128
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)


with tf.device('/gpu:0'):
    predictions = model4.predict(X_train)
    print('Предсказание: ', predictions)
    group_code = np.where(predictions == np.amax(predictions))
    print('Код группы: ', group_code)








