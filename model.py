import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('output2.csv')

print(df.columns)

df.fillna({ 'stemmed_content': ''}, inplace=True)

print(df.isnull().sum())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['stemmed_content'])
word_index = tokenizer.word_index

df['tokenized_text'] = tokenizer.texts_to_sequences(df['stemmed_content'])
print(df['tokenized_text'])

padded_sequences = pad_sequences(df['tokenized_text'], maxlen=100, padding='post', truncating='post')

num_classes = 6  # 0 to 5 sentiments

one_hot_labels = to_categorical(df['label'], num_classes=num_classes)

model = Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

batch_size = 128
epochs = 1  # You can adjust this based on your needs
model.fit(padded_sequences, one_hot_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

# Testing the model on a single input
# Example usage for prediction
new_texts = ["I love this product!", "Terrible service."]
# Convert to DataFrame
df_input = pd.DataFrame(new_texts, columns=["Text"])
tokenizer.fit_on_texts(df_input["Text"])

new_sequences = tokenizer.texts_to_sequences(df_input["Text"])
print(new_sequences)

padded_sequences_input = pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')
print(model.predict(padded_sequences_input))

# To save the model
model.save('/home/shreyansh/ml-project/pythonProject/saved_model_1.keras')
