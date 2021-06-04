from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    BatchNormalization,
    Flatten,
    Bidirectional,
    LSTM,
    Dropout,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import re


# grab the text
data = fetch_20newsgroups(remove=["headers", "footers"])
text = data["data"]

# create a continuous list of words
all_data = " ".join(
    [" ".join(re.findall("(?u)\\b[a-zA-Z]*\\b", article.lower())) for article in text]
).split()

# work out the vocab list and size of vocab
vocab_list = sorted(list(set(all_data)))
n_vocab = len(vocab_list)

# translate words to numbers
word_to_num = {}
num_to_word = {}
for i, word in enumerate(vocab_list):
    num_to_word[i] = word
    word_to_num[vocab_list[i]] = i

# embed the data
embedded_data = [word_to_num[word] for word in all_data]

# create the next word guess for each previous 10 words
X_data = []
y_data = []
seq_length = 10
for i in range(len(embedded_data) - seq_length):
    X_data.append(embedded_data[i : i + seq_length])
    y_data.append(embedded_data[i + seq_length])


# reshape the X and y data
X = np.array(X_data).reshape(len(X_data), seq_length)
y = to_categorical(y_data)


# build the model
model = Sequential()
model.add(Embedding(input_dim=n_vocab, output_dim=32, input_length=seq_length))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True), merge_mode="sum"))
model.add(LSTM(128))
model.add(
    BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
)
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# save best version of the model
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

# fit the model
model.fit(X, y, epochs=20, batch_size=256, validation_split=0.2, callbacks=callbacks)
