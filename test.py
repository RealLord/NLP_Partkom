import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalMaxPool1D, Conv1D
from keras.layers import Embedding

def learn(dataset):
    df_draft = pd.read_csv(dataset, sep=';', header=0, encoding='utf-8',
                           dtype={' group_code': int, ' desc': str}, compression="zip", usecols=[2, 4])
    df_draft = df_draft.loc[df_draft[' group_code'] != 0]
    sentences = df_draft[' desc']
    y = df_draft[' group_code']

    tokenize = Tokenizer(num_words=5000)
    tokenize.fit_on_texts(sentences)

    X_train = tokenize.texts_to_sequences(sentences)
    vocab_size = len(tokenize.word_index) + 1

    maxlen = 25
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

    encoder = LabelEncoder()
    encoder.fit(y)
    # with open("data/group_code_mapping_test.txt", "w", encoding="utf-8") as o:
    #     for i, key in enumerate(encoder.classes_):
    #         o.write(str(i) + " " + str(key) + "\n")


    y_train = encoder.transform(y)
    #y_test = encoder.transform(y)
    num_classes = np.max(y_train) + 1
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    #y_test  = tf.keras.utils.to_categorical(y_test,  num_classes)


    embedding_dim = 100
    model4 = Sequential()
    model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model4.add(Flatten())
    model4.add(Dense(10000, activation='relu'))
    model4.add(Dense(num_classes, activation='softmax'))

    model4.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    model4.summary()

    history_4 = model4.fit(X_train, y_train,
                           batch_size=256,  # 32 - default
                           epochs=1)  # 15 - default
                           # validation_data=(X_test, y_test))
    print(history_4)

if __name__ == "__main__":
    dataset = 'data/goods_out_v2.zip'
    learn(dataset)