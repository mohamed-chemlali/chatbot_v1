import nltk
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import json
import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)

except:    
    words = []
    labels = []
    docs = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]: # intent is a hole tag with his pattern
        for pattern in intent["patterns"]: # patern is a sentence in patterns
            wrds = nltk.word_tokenize(pattern) # wrds is list contains each word in pattern splitted by the word_tokenize() function
            words.extend(wrds) # add each word in wrds to words variable
            docs_x.append(wrds)  #
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])  

    words = [st.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [st.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:    
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_word(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [st.stem(word.lower()) for word in s_words]
     
    for se in s_words:
         for i, w in enumerate(words):
             if w == se:
                 bag[i] = 1

    return np.array(bag)

def chat():
    print("start talking with the bot!")
    while True:
        inp = input("You:")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_word(inp, words)]) 
        results_index = np.argmax(results)
        tag = labels[results_index]
        

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))        

chat()
