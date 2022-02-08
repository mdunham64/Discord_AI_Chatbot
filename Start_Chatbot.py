import discord
import nltk
from nltk.stem.lancaster import LancasterStemmer

#######################################################################################################
######################                IMPORT OUR NETWORK           ####################################
######################                                             ####################################
#######################################################################################################


stemmer = LancasterStemmer()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tflearn
import random
import wikipedia
import pickle

lang = "en"

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json

with open('intents.json', encoding='utf8') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


#######################################################################################################
######################          CLEAN-UP INPUT LANGUAGE            ####################################
######################                                             ####################################
#######################################################################################################


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return np.array(bag)


# load our saved model
model.load('./model.tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25


#######################################################################################################
######################             GENERATE PREDICTION             ####################################
######################                                             ####################################
#######################################################################################################


def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


#######################################################################################################
######################                GENERATE RESPONSE            ####################################
######################                                             ####################################
#######################################################################################################


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print('tag:', i['tag'])
                        val = random.choice(i['responses'])
                        # a random response from the intent
                        return val

            results.pop(0)


print("Bot is running.")
client = discord.Client()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!aibot"):  # if the discord bot sees its command
        final_response = ""
        question = message.content[7:]  # index 7 and onwards because we want to ignore the bot command in our Q
        question = str(question)  # make sure its a string
        final_response = response(question)  # generate our response

        if 'google' in question or 'Google' in question:
            filtered_question = question
            if 'Google' in filtered_question:
                filtered_question = filtered_question.split("Google")[1].replace(" ", "+")
            if 'google' in filtered_question:
                filtered_question = question.split("google")[1].replace(" ", "+")

            final_response = f"Here is what I could find for you... \n\n https://google.com/search?q={filtered_question}"

        if 'wiki' in question or 'wikipedia' in question:
            try:
                final_response = wikipedia.search(str(question))
            except wikipedia.exceptions.PageError:
                final_response = "Uh oh!! I don't understand you... Can you please rephrase your wikipedia search? Try 'Wiki [search term]'. "

        if final_response == "":
            final_response = question

        await message.channel.send(final_response)


token = "insert_DiscordBot_token_here"
client.run(token)
