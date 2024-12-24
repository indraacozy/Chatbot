import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_word = ['?','!']
data_file = open("models\intents_final.json").read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_word]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lematized words", words)

pickle.dump(words,open('text_test.pkl','wb'))
pickle.dump(classes,open('labels_test.pkl','wb'))

training = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_word = doc[0]

    pattern_word = [lemmatizer.lemmatize(word.lower()) for word in pattern_word]

    for w in words:
        bag.append(1) if w in pattern_word else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

print('Training data complete!')


model = Sequential()
model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(254, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=400, batch_size=5, verbose=1)
model.save('chatbot_model_test.h5', hist)
print("\n")
print("*" * 50)
print("\n Model Created Complete!")
