#%% Step 1: Generate corpus for each language

train_ita = data_train.loc[data_train['lang']=='ita']['sent']
train_deu = data_train.loc[data_train['lang']=='deu']['sent']
train_por = data_train.loc[data_train['lang']=='por']['sent']
train_eng = data_train.loc[data_train['lang']=='eng']['sent']
train_spa = data_train.loc[data_train['lang']=='spa']['sent']
train_fra = data_train.loc[data_train['lang']=='fra']['sent']

train_ita.to_csv(r'/Users/Cathy/Documents/code/ML-LD/ita.txt', header=None, index=None)
train_deu.to_csv(r'/Users/Cathy/Documents/code/ML-LD/deu.txt', header=None, index=None)
train_por.to_csv(r'/Users/Cathy/Documents/code/ML-LD/por.txt', header=None, index=None)
train_eng.to_csv(r'/Users/Cathy/Documents/code/ML-LD/eng.txt', header=None, index=None)
train_spa.to_csv(r'/Users/Cathy/Documents/code/ML-LD/spa.txt', header=None, index=None)
train_fra.to_csv(r'/Users/Cathy/Documents/code/ML-LD/fra.txt', header=None, index=None)

test_lang = data_test['lang']
test_sent = data_test['sent']
test_lang.to_csv(r'/Users/Cathy/Documents/code/ML-LD/test_lang.txt', header=None, index=None)
test_sent.to_csv(r'/Users/Cathy/Documents/code/ML-LD/test_sent.txt', header=None, index=None)


#%% Step 2: Corpus preprocessing 

# See @ process.py 
# Command line @ n-gram.sh

'''
# Let's start a prototype

import sys
import nltk

filename = '/Users/Cathy/Documents/code/ML-LD/ita.txt'
file = open(filename,'r')
tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`éèî]+") # The re expression needs to be edited

for i,line in enumerate(file):
    if i < 100:
        for sentence in nltk.sent_tokenize(line):
            print(' '.join(tokenizor.tokenize(sentence)).lower())
   
print(i,line)
    
sent = "This is my text, this is a nice way to input text. Hell0! Love you."
nltk.sent_tokenize(sent)
print(' '.join(tokenizor.tokenize(sent)).lower())
'''


#%% Step 3: Model compile

# Command line @ model.sh
