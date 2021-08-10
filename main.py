# abstract class
from abc import ABCMeta, abstractmethod

# utlis 
import random
import json
import pickle
import numpy as np
import os
import logging
from tqdm import tqdm

# NLP and Neural network module
import nltk
from nltk.stem import WordNetLemmatizer # get the synonyms (ex. works, working, worked...)
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Softmax
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

logging.basicConfig(level=logging.INFO)

nltk.download('punkt', quiet=True) # use to tokenize
nltk.download('wordnet', quiet=True)# 
class GenericAssistant(object):

    def __init__(self, intents, intent_methods={}, model_name="assistant_model"):
        super().__init__()
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name

        if intents.endswith(".json"): # load training data
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def get_training_dataset(self):
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag'].replace('_', ' ')))
                if intent['tag'].replace('_', ' ') not in self.classes:
                    self.classes.append(intent['tag'].replace('_', ' '))
                
        
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))
        
        self.classes = sorted(list(set(self.classes)))
        logging.debug(f'{self.classes}\n{documents}\n{self.words}\n{self.classes}')

        training = []

        for doc in documents:
            bag = []
            word_patterns = doc[0] #get word patterns
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words: # bag of word
                bag.append(1) if word in word_patterns else bag.append(0)

            label = self.classes.index(doc[1]) # get label
            training.append([bag, label])

        random.shuffle(training)
        training = np.array(training)
        
        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])
        #logging.debug(f"{train_x, train_y}")
        

    def set_dataloader(self, batch_sz):
        train_x, train_y = torch.tensor(self.train_x, dtype=torch.float), torch.LongTensor(self.train_y)
        train_data = TensorDataset(train_x, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_sz)
        return train_dataloader

    def get_model(self):
        model = Sequential(
            Linear(len(self.train_x[0]), 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 64),
            ReLU(),
            Dropout(0.5),
            Linear(64, len(self.classes))
        )
        '''
        optimizer = SGD(model.parameters(),
                      lr=8e-4,    # Default learning rate
                      momentum=0.9,    # Default epsilon value
                      weight_decay=1e-6,
                      nesterov=True
                      )
        '''
        optimizer = AdamW(model.parameters(),
                      lr=6e-5,    # Default learning rate
                      eps=1e-8,
                      betas=(0.9, 0.999)
                      )
        
        return model, optimizer
        
    def train_model(self, epoch=5):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        for e in range(epoch):
            epoch_loss = 0
            for step, (x, y) in enumerate(self.train_dl):
                output = self.model(x)
                loss = loss_fn(output, y.view(-1))
                epoch_loss += loss
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            if (e+1) % 25 == 0:    
                logging.info(f'E_{e+1} Epoch_loss：{epoch_loss/len(self.train_dl)}')
        
    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.model_name}.h5")
        pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
        pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))
    
    def load_model(self):
        self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
        self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))
        self.model, self.optimizer = self.get_model()
        checkpoint = torch.load(f"{self.model_name}.h5")
        self.model.load_state_dict(checkpoint)

    def run(self):
        self.get_training_dataset()
        self.train_dl = self.set_dataloader(5)
        self.model, self.optimizer = self.get_model()
        logging.debug(self.model)
        self.train_model(250)

if __name__ == '__main__':
    tmpAs = GenericAssistant("intents.json")
    tmpAs.run()