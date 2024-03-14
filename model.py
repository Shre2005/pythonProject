import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('output2.csv')

df.fillna({ 'stemmed_content': '' }, inplace=True)

print(df.isnull().sum())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['stemmed_content'])
word_index = tokenizer.word_index

df['tokenized_text'] = tokenizer.texts_to_sequences(df['stemmed_content'])
print(df['tokenized_text'])

padded_sequences = pad_sequences(df['tokenized_text'], maxlen=100, padding='post', truncating='post')

num_classes = 6  # 0 to 5 sentiments

one_hot_labels = to_categorical(df['label'], num_classes=num_classes)
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(len(word_index) + 1, 100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(padded_sequences, one_hot_labels, epochs=50, validation_split=0.2, callbacks=[early_stopping])

