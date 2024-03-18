import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2

#column_names = ["target", "id", "date", "flag", "user", "text"]
new_data = pd.read_csv("subset_dataset4.csv", encoding="ISO-8859-1")
'''',  names=column_names , header=1)'''

words = set()
for data in new_data['text']:
    # Check if the value is a string
    if isinstance(data, str):
        # Split each text data into words
       print(words.update(data.split()))

print(len(words))

new_data = new_data.dropna(subset=['text'])

max_features = len(words)
tokenizer_keras = Tokenizer(num_words=max_features, split=' ')
tokenizer_keras.fit_on_texts(new_data['text'])
X = tokenizer_keras.texts_to_sequences(new_data['text'])
X = pad_sequences(X)
y = pd.get_dummies(new_data['target']).values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

validation_size = 150000

X_validate = X_test[-validation_size:]
y_validate = y_test[-validation_size:]
X_test = X_test[:-validation_size]
y_test = y_test[:-validation_size]

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import Adam


embed_dim = 128
lstm_out = 196
'''''
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001)))
optimizer = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])'''''


model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001)))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

print(model.summary())

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",mode="min", patience=5,restore_best_weights=True)
model.fit(X_train, y_train, epochs = 2 , verbose = 2, validation_data=(X_validate, y_validate),callbacks=[earlystopping])

#model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=1, validation_data=(X_validate, y_validate), callbacks=[earlystopping])


model.save('saved_model_final3.keras')

score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=64)
print("Test score: %.2f" % (score))
print("Test accuracy: %.2f" % (acc))

y_pred = model.predict(X_test)
cfm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), normalize='pred')
print("Confusion Matrix:")
print(cfm)












'''''
data = df[["target", "stemmed_content"]]
print(data)
print(df.columns)

df.fillna({'stemmed_content': ''}, inplace=True)

print(df.isnull().sum())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['stemmed_content'])
word_index = tokenizer.word_index

df['tokenized_text'] = tokenizer.texts_to_sequences(df['stemmed_content'])
print(df['tokenized_text'])

padded_sequences = pad_sequences(df['tokenized_text'], maxlen=100, padding='post', truncating='post')

# Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)
num_classes = 2 # 0 to 5 sentiments

one_hot_labels = to_categorical(df['target'], num_classes=num_classes)

model = Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

epochs = 1  # You can adjust this based on your needs
batch_size = 128
model.fit(padded_sequences, one_hot_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

# To save the model
model.save('saved_model_final2.keras')'''''
