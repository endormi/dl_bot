# dl_bot

Deep learning chatbot (Using **Python 3.6.1**)

Needed libraries:

```python
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import tensorflow as tf
import tflearn
import numpy
import random
import json
```

Install requirements:

```sh
pip install -r requirements.txt
```

Use `tflearn` to train model:

```python
net_neuron = tflearn.input_data(shape=[None, len(train[0])])
net_neuron = tflearn.fully_connected(net_neuron, 8)
net_neuron = tflearn.fully_connected(net_neuron, 8)
net_neuron = tflearn.fully_connected(net_neuron, len(output[0]), activation="softmax")
net_neuron = tflearn.regression(net_neuron)
model_layer = tflearn.DNN(net_neuron)
model_layer.fit(train, output, n_epoch=1500, batch_size=8, show_metric=True)
model_layer.save("model_layer.tflearn")
```

**Additional**: With `nltk` you might need to install `punkt`:

Go to **Python shell** and type:

```sh
>>> import nltk
>>> nltk.download()
```