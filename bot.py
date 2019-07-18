import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import tensorflow
import tflearn
import numpy
import random
import json

stem = LancasterStemmer()


with open("data.json") as file:
    messages = json.load(file)


# Empty lists to loop through
words = []
labels = []
doc = []
diff_doc = []

# Loop through dictionaries in data
for message in messages["message"]:
    for pat in message["patterns"]:
        """
        Stem words with nltk's tokenizer,
        get all the words in a pattern
        Add tokenized words in words list
        """
        many_words = nltk.word_tokenize(pat)
        words.extend(many_words)
        doc.append(many_words)
        diff_doc.append(message["tag"])

        if message["tag"] not in labels:
            labels.append(message["tag"])


"""
Stem words list and remove duplicates,
convert words to lowercase
set a list of words
"""
words = [stem.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

"""
List of words to represent all of the words in any given pattern,
to train model
"""
train = []
output = []
out_empty = [0 for _ in range(len(labels))]


for x, res in enumerate(doc):
    list_of_words = []
    many_words = [stem.stem(w) for w in res]

    """
    Loop through the words that are in list of words
    and put 1 depending on if the word exists
    """
    for w in words:
        if w in many_words:
            list_of_words.append(1)
        else:
            list_of_words.append(0)

    # See where "tag" is and set value to 1
    output_row = out_empty[:]
    output_row[labels.index(diff_doc[x])] = 1

    train.append(list_of_words)
    output.append(output_row)

train = numpy.array(train)
output = numpy.array(output)


tensorflow.reset_default_graph()

# Define input shape expected for model
net_neuron = tflearn.input_data(shape=[None, len(train[0])])
# Use 8 neurons for hidden layer
net_neuron = tflearn.fully_connected(net_neuron, 8)
net_neuron = tflearn.fully_connected(net_neuron, 8)
# Allow to get probabilities for each neuron in layer
net_neuron = tflearn.fully_connected(net_neuron, len(output[0]), activation="softmax")
net_neuron = tflearn.regression(net_neuron)


model_layer = tflearn.DNN(net_neuron)
# n_epoch is the amount of time it's going to see the same data
model_layer.fit(train, output, n_epoch=1500, batch_size=8, show_metric=True)
model_layer.save("model_layer.tflearn")

