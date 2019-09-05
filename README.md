# dl_bot

[![Python Version](https://img.shields.io/badge/python-3.6.1-brightgreen.svg?)](https://www.python.org/downloads/)

> Newer versions of python might not work.

Deep learning chatbot.

Install requirements:

```sh
pip install -r requirements.txt
```

**Additional**: With `nltk` you might need to install `punkt`:

Go to **Python shell** and type:

```sh
>>> import nltk
>>> nltk.download()
```

The bots answers are based on `data.json`, which has different patterns.

I have included five tags "greetings", "goodbye", "age", "name" and "robot".

Talking with the bot:

```
"greetings":
"Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up"

"goodbye":
"cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "quit", "bye"

"age":
"how old", "what is your age", "how old are you", "age?"

"name":
"what is your name", "what should I call you", "whats your name?"

"robot":
"bot", "do robot stuff", "robot"
```

Be sure to test this out!
