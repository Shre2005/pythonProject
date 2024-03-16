import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#model = tf.saved_model.load("/home/shreyansh/ml-project/pythonProject/saved_model")
model = load_model("saved_model_1.keras")


#print(type(model))
#print("Signatures:", model.signatures)

tokenizer = Tokenizer()
# Example usage for prediction
new_texts = ["I love this product!", "Terrible service."]
# Convert to DataFrame
df = pd.DataFrame(new_texts, columns=["Text"])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["Text"])


new_sequences = tokenizer.texts_to_sequences(df["Text"])
print(new_sequences)

padded_sequences = pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')
"""
predictions = model.predict(padded_sequences)

# Output predictions
for i, text in enumerate(new_texts):
    predicted_class = np.argmax(predictions[i])
    print(f"Text: {text}\nPredicted Sentiment Class: {predicted_class}\n")
"""
