import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import random
import json

stem = LancasterStemmer()


with open("data.json") as file:
    messages = json.load(file)

print(messages["message"])