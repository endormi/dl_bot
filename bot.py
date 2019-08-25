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

try:
    """
    Load in pickle data for model
    containing words, labels, train and output
    if data is already saved and processed,
    no reason to do it again
    """
    with open("data.pickle", "rb") as f:
        words, labels, train, output = pickle.load(f)
except:
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


    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, train, output), f)


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


try:
    model_layer.load("model_layer.tflearn")
except:
    # n_epoch is the amount of time it's going to see the same data
    model_layer.fit(train, output, n_epoch=1500, batch_size=8, show_metric=True)
    model_layer.save("model_layer.tflearn")


def words_list(s, words):
    list_of_words = [0 for _ in range(len(words))]

    each_words = nltk.word_tokenize(s)
    each_words = [stem.stem(many_words.lower()) for many_words in each_words]

    for r in each_words:
        for i, w in enumerate(words):
            """
            If current word in list is equal to word 
            in sentence, it will append by 1
            """
            if w == r:
                list_of_words[i] = 1

    return numpy.array(list_of_words)


# Ask user for a sentence
def chat_with_bot():
    print("Start talking with the chatbot (type close to close the chat)!")
    while True:
        add_input = input("You: ")
        if add_input.lower() == "close":
            break
        result = model_layer.predict([words_list(add_input, words)])
        # Index of the greatest value in list
        result_index = numpy.argmax(result)
        tag = labels[result_index]
        

        for answer in messages["message"]:
            if answer["tag"] == tag:
                responses = answer["responses"]
        print(random.choice(responses))

chat_with_bot()