#%% Test with English sentences

import nltk
import langdetect

# Test package
langdetect.detect('Hello,How are you?')
# copy a test dataset 
data = data_valid.copy()
data.sent = data.sent.str.lower() # lowercase
tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`éèî]+") # this re need to be edited
token = list()
trigram = list()
for i in range(len(data)):
    token.append(tokenizor.tokenize(data['sent'][i+1]))
    trigram.append(nltk.trigrams(token[i]))
