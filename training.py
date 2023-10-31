import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping

# Schritte 1: Daten sammeln und vorbereiten (Beispieltext)
text_corpus = [

    '''
    schade.

    verstehe.
    ''',
'''
was bedeutet wetter?

Wetter ist der Zustand der Atmosphäre an einem bestimmten Ort und Zeitpunkt.
Es wird durch verschiedene Faktoren wie Temperatur, Luftdruck, Luftfeuchtigkeit, Niederschlag, Wind und Wolken beeinflusst.
 Wetter kann sich schnell ändern und hat Auswirkungen auf Menschen, Tiere, Pflanzen und Umwelt.
''',
'''
was ist internet?

Das Internet ist ein weltweites Netzwerk von Computern, die miteinander kommunizieren können.
Das Internet ermöglicht den Austausch von Informationen, Daten, Bildern, Videos, Musik und vielem mehr.
Das Internet hat viele Anwendungen, wie zum Beispiel E-Mail, Webseiten, Online-Shopping, Soziale Medien,
 Online-Spiele und vieles mehr. Das Internet ist eine wichtige Quelle für Wissen, Unterhaltung und Zusammenarbeit.
''',
'''
Wie wird das Wetter morgen?

Es tut mir Leid, aber ich habe kein Internet Zugriff.
''',

    '''
sdfa wehf fxsdawe whe sdf

Es tut mir leid, aber ich verstehe die frage nicht.
        ''',
#erweitere daten...
]

# Schritte 2: Tokenisierung und Sequenzierung der Textdaten
tokenizer = Tokenizer(char_level=True)
tokenizer = Tokenizer()
tokenizer = Tokenizer(filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(text_corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
lstm_layer = tf.keras.layers.LSTM(units=550, return_sequences=True, activation='tanh')
lstm_layer = tf.keras.layers.LSTM(units=550, return_sequences=True, activation='tanh')


# Schritte 3: Modell erstellen und trainieren
model = keras.Sequential([
    keras.layers.Embedding(total_words, 650, input_length=max_sequence_length - 1),
    lstm_layer,


    tf.keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Bidirectional(keras.layers.LSTM(550,  activation='tanh')),


 # LSTM mit Gedächtnis (return_sequences=True) hinzufügen

    keras.layers.Dense(550, activation='relu'),# NLU-Schicht hinzufügen
    keras.layers.Dense(550, activation='selu'), # stablier
keras.layers.Dense(550, activation='gelu'),
    keras.layers.Dense(total_words, activation='softmax'),  # NLG-Schicht hinzufügen
])





model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=1 , verbose=1, batch_size=200)

import os

# Erstellen Sie einen Ordner für das Modell und speichern Sie das Modell und den Tokenizer darin
model_dir = 'FT'
os.makedirs(model_dir, exist_ok=True)

# Speichern Sie das Modell und den Tokenizer im "modell"-Ordner
model.save(os.path.join(model_dir, 'model.h5'))

# Speichern Sie den Tokenizer im "modell"-Ordner
tokenizer_json = tokenizer.to_json()
with open(os.path.join(model_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)


# Speichern Sie die max_sequence_length in einer Datei "max_len.txt"
with open(os.path.join(model_dir, 'max_len.txt'), 'w') as f:
    f.write(str(max_sequence_length))
