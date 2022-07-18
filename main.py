from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras.models import Sequential
from keras.metrics import Metric
from keras.layers import Dense, Dropout, GlobalMaxPool1D
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.function(jit_compile=True)  # Включение XLA
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('ERROR: GPU exception')
    exit(-100)


##################
debug = True
##################

########## Config ############
num_words = 3000000
maxlen = 50
embedding_dim = 30
epochs = 5
batch = 32
desc_str = "metrics_categorical_accuracy"
##############################

now = datetime.now()
df_draft = pd.read_csv('data/Final/100_final_not_0_group_more_350_filtered_without_art_brend_2char_dedup_cleaned_stop_cutted.csv', sep=';', header=0, encoding='utf-8', dtype={"desc": "string", "group_code": int})
later = datetime.now()
difference = (later - now).total_seconds()

if debug:
    print("01: Dataframe loaded!, %s" % difference)

sentences = df_draft['desc']
y = df_draft['group_code']
maxgroups = y.value_counts().size
print('02: Unique group codes: %s' % maxgroups)
print('03: Max lenght desc: %s' % sentences.str.len().max())

sentences_train, sentences_test, train_y, test_y = train_test_split(sentences, y, test_size=0.30, stratify=y)
print('04: Split train and test')


now = datetime.now()
with tf.device('/cpu:0'):
    tokenize = Tokenizer(num_words=num_words)  # CHANGE
    tokenize.fit_on_texts(pd.concat([sentences_train], axis=0).astype("str"))
later = datetime.now()
difference = (later - now).total_seconds()
print('05.10: Tokenizer.fit_on_texts(): %s' % difference)


now = datetime.now()
with tf.device('/cpu:0'):
    X_train = tokenize.texts_to_sequences(pd.concat([sentences_train], axis=0).astype("str"))
    X_test = tokenize.texts_to_sequences(pd.concat([sentences_test], axis=0).astype("str"))
    vocab_size = len(tokenize.word_index) + 1
later = datetime.now()
difference = (later - now).total_seconds()
print('05.20: tokenize.texts_to_sequences: %s' % difference)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print('05.00: Made tokenization')

# вариант с энкодингом
encoder = LabelEncoder()
encoder.fit(train_y)

#################### Пишем список group_code на диск для последующего анализа #############################
with open("data/group_code_mapping_test.txt", "w", encoding="utf-8") as o:
     for i, key in enumerate(encoder.classes_):
         o.write(str(i) + " " + str(key) + "\n")
###########################################################################################################

y_train = encoder.transform(train_y)
y_test = encoder.transform(test_y)
num_classes = np.max(y_train) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes)
print('06: Made label encoding and write groups to file')

model4 = Sequential()
model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model4.add(GlobalMaxPool1D())
model4.add(Dropout(0.2))  # CHANGE
model4.add(Dense(maxgroups*5, activation='relu'))  #16 is default  # CHANGE
model4.add(Dropout(0.2))  # CHANGE
model4.add(Dense(maxgroups, activation='softmax'))  # 431 - number of unique group_code

model4.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
#                metrics=['accuracy'])
                metrics=['categorical_accuracy']
)

model4.summary()
with tf.device('/gpu:0'):
    history_4 = model4.fit(X_train, y_train,
                          batch_size=batch,  # 32 - default # CHANGE
                          epochs=epochs,  # 15 - default
                          validation_data=(X_test, y_test))

    s = f'{datetime.now():%Y_%m_%d_%H_%M}'
    model_name = 'model_acc_'+str(history_4.history['val_accuracy'][0])[:5]+'_loss_'+str(history_4.history['val_loss'][0])[:5]+'_nw_'+str(num_words)+'_maxl_'+str(maxlen)+'_emb_'+str(embedding_dim)+'_batch_'+str(batch)+'_epochs_'+str(epochs)+'_desc_'+desc_str+'.'+s
    model4.save(model_name)  # Saving model!
    #################### Пишем список group_code на диск для последующего анализа #############################
    with open(model_name+"/group_code_mapping_test.txt", "w", encoding="utf-8") as o:
         for i, key in enumerate(encoder.classes_):
             o.write(str(i) + " " + str(key) + "\n")
    ###########################################################################################################


#if debug:
#      print('Go to evaluate')
#
with tf.device("/gpu:0"):
    fo1 = open(model_name+"/evaluate_result.txt", 'w', encoding='utf-8')
    loss, accuracy = model4.evaluate(X_train, y_train, verbose=False, batch_size=16)
    s = "Training Accuracy: {:.4f}".format(accuracy)
    print(s)
    fo1.writelines(s)
    s = "Training loss: {:.4f}".format(loss)
    print(s)
    fo1.writelines(s)

    loss_test, accuracy_test = model4.evaluate(X_test, y_test, verbose=False, batch_size=16)
    s = "Testing Accuracy:  {:.4f}".format(accuracy_test)
    print(s)
    fo1.writelines(s)
    s = "Testing loss:  {:.4f}".format(loss_test)
    print(s)
    fo1.writelines(s)

###################################### CNN #######################################
# CPU = 40 минут


# model5 = Sequential()
# model5.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
# model5.add(Conv1D(128, 5, activation='relu'))  #  128 5 default
# model5.add(GlobalMaxPool1D())
# model5.add(Dense(maxgroups*5, activation='relu'))  # 10 default
# model5.add(Dense(maxgroups, activation='softmax'))
#
# model5.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
# model5.summary()
#
# with tf.device("/gpu:0"):
#   history_5 = model5.fit(X_train, y_train,
#                   batch_size = batch,  # 32 - default # CHANGE
#                   epochs = epochs,  # 15 - default
#                  validation_data=(X_test, y_test))
#
# s = f'{datetime.datetime.now():%Y_%m_%d_%H_%M}'
# model5.save('model5.save.'+s)  # Saving model!
#
# if debug:
#     print('Go to evaluate')
#
# with tf.device("/cpu:0"):
#     loss, accuracy = model5.evaluate(X_train, y_train, verbose=False)
#     print("Training Accuracy: {:.4f}".format(accuracy))
#     loss, accuracy = model5.evaluate(X_test, y_test, verbose=False)
#     print("Testing Accuracy:  {:.4f}".format(accuracy))


