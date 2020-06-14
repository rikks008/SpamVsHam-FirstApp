#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[21]:


data = pd.read_csv("spam.csv",encoding='latin-1')


# In[23]:


data.head()


# In[24]:


data.dropna(inplace=True,axis=1)
data.columns=['label','message']


# In[26]:


data.head()


# In[27]:


data.info()


# In[28]:


data.describe()


# In[29]:


data['label'].value_counts()


# ## Text Preprocessing
# 

# In[40]:



import unicodedata
import re
import nltk
import numpy as np

ps = nltk.porter.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[31]:


## remove special charactors
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

## Remove stop words

def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

## Stemming
def simple_stemming(text, stemmer=ps):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

## REmove accented charactors
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# In[34]:


def pre_process_document(document):
    
    document = document.lower()
    
    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))
    
    # remove accented characters
    document = remove_accented_chars(document)
    
    # remove special characters and\or digits    
    # insert spaces between special characters to isolate them    
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=True)  

    # stemming text
    document = simple_stemming(document)      
    
    # remove stopwords
    document = remove_stopwords(document, is_lower_case=True, stopwords=stop_words)
        
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
    
    return document


pre_process_corpus = np.vectorize(pre_process_document)


# In[35]:


data['clean_text']= pre_process_corpus(data['message'])


# In[36]:


data.head()


# In[37]:


# convert label to a numerical variable
data['label'] = data.label.map({'ham':0, 'spam':1})


# In[41]:


data.head()


# ### Converting text into numbers

# In[54]:


max_features = 10000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['clean_text'])


# In[55]:


tokenizer.document_count


# In[56]:


len(tokenizer.word_index)


# In[57]:


# get the length of each message and find the max length
for i in range(len(data['clean_text'])):
  length = len(data.loc[i,'clean_text'])
  if(length > maxlen):
    maxlen = length


# In[58]:


maxlen


# In[61]:


X =tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(X, maxlen = maxlen)
print("Number of Samples:", len(X))
print(X[0])
y = np.asarray(data['label'])
print(y[0])


# In[63]:


num_words = len(tokenizer.word_index) + 1
num_words


# ## Building the model

# In[87]:


from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
from numpy import asarray
from numpy import zeros
from keras.optimizers import RMSprop


# In[78]:


glove_file = 'glove.6B.200d.txt'

embeddings_index = dict()
f = open(glove_file)
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# embedding_size=200
# vocab_size=10000
# max_len=300
# model = Sequential() 
# model.add(Embedding(num_words,embedding_size,max_len)) 
# model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
# model.add(Dense(15,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1,activation='sigmoid'))
# 
#  compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#  summarize the model
# print(model.summary()) 

# In[94]:


max_words = 10000
max_len = 427
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[95]:


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[96]:


batch_size = 100
epochs = 10
model.fit(X,y,batch_size=batch_size,epochs=epochs,validation_split=0.2)


# In[99]:


# Creating a pickle file for the Multinomial Naive Bayes model
import pickle
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))


# In[100]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-ham-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




