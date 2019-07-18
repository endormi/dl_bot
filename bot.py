import nltk
from nltk.stem.lancaster import LancasterStemmer
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
        doc.append(pat)
        diff_doc.append(message["tag"])

        if message["tag"] not in labels:
            labels.append(message["tag"])


"""
Stem words list and remove duplicates,
convert words to lowercase
set a list of words
"""
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
labels = sorted(labels)

"""
List of words to represent all of the words in any given pattern,
to train model
"""
train = []
output = []
out_empty = [0 for _ in range(len(classes))]


for x, res in enumerate(doc):
    list_of_words = []
    many_words = [stemmer.stem(w) for w in res]

    """
    Loop through the words that are in list of words
    and put 1 depending on if the word exists
    """
    for w on words:
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
output = np.array(output)