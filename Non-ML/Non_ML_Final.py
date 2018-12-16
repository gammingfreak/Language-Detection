#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## N-gram generating

#%% Step0: Loading Data - Training Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load three datasets
data_train = pd.read_csv('~/Downloads/train.csv',sep=',', encoding='utf8', header = None)
data_valid = pd.read_csv('~/Downloads/valid.csv',sep=',', encoding='utf8', header = None)
data_test = pd.read_csv('~/Downloads/test.csv',sep=',', encoding='utf8', header = None)

# Drop the head & first column of three datasets 
data_train.drop([0],axis=0,inplace = True)
data_valid.drop([0], axis=0,inplace=True)
data_test.drop([0],axis=0,inplace = True)
data_train.drop([0], axis=1,inplace=True)
data_valid.drop([0], axis=1,inplace=True)
data_test.drop([0], axis=1,inplace=True)

# Set the name of column
data_train.columns = ['lang','sent']
data_valid.columns = ['lang','sent']
data_test.columns = ['lang','sent']

# Size of each dataset
n_lang = 6 # number of languages
n_train = len(data_train)
n_valid = len(data_valid)
n_test = len(data_test)

# Distribution of string length
str_len_train = data_train.sent.str.len()
plt.hist(str_len_train)

print(data_train['lang'].unique()) # the language we keep: ['ita' 'deu' 'por' 'eng' 'spa' 'fra']
data_train.head()

#%% Step 1: Generate corpus for each language

train_ita = data_train.loc[data_train['lang']=='ita']['sent']
train_deu = data_train.loc[data_train['lang']=='deu']['sent']
train_por = data_train.loc[data_train['lang']=='por']['sent']
train_eng = data_train.loc[data_train['lang']=='eng']['sent']
train_spa = data_train.loc[data_train['lang']=='spa']['sent']
train_fra = data_train.loc[data_train['lang']=='fra']['sent']

train_ita.to_csv(r'~/Documents/code/ita.txt', header=None, index=None)
train_deu.to_csv(r'~/Documents/code/deu.txt', header=None, index=None)
train_por.to_csv(r'~/Documents/code/por.txt', header=None, index=None)
train_eng.to_csv(r'~/Documents/code/eng.txt', header=None, index=None)
train_spa.to_csv(r'~/Documents/code/spa.txt', header=None, index=None)
train_fra.to_csv(r'~/Documents/code/fra.txt', header=None, index=None)

test_lang = data_test['lang']
test_sent = data_test['sent']
test_lang.to_csv(r'~/Documents/code/test_lang.txt', header=None, index=None)
test_sent.to_csv(r'~/Documents/code/test_sent.txt', header=None, index=None)


#%% Step 2: Corpus preprocessing 

# See @ process.py 
# Command line @ n-gram.sh
# It will generate 6 arpa file - each contains character-based tri-gram.

#%% Step 3: Model compile

# Command line @ model.sh
# It will generate an n-gram model for each language - klm

#%% Step 4: Testing dataset preprocessing

# See @ test_process.py
# Command line @ test_process.sh
# After processing, the new testing dataset is test_set_new.txt

#%% Step 4: Load Model and score sentences

import kenlm

with open('/Users/yifang/Documents/code/test_set_new.txt') as f:
    test_set_new = f.readlines()

# Load model 
model_ita = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/ita.klm')
model_fra = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/fra.klm')
model_por = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/por.klm')
model_eng = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/eng.klm')
model_spa = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/spa.klm')
model_deu = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/deu.klm')

# Calculate score & preplextity
score = np.full([n_test,n_lang],np.nan)
perplexity = np.full([n_test,n_lang],np.nan)

for i in range(n_test):
    score[i,0] = model_eng.score(test_sent_new[i])
    perplexity[i,0] = model_eng.perplexity(test_sent_new[i])
    score[i,1] = model_deu.score(test_sent_new[i])
    perplexity[i,1] = model_deu.perplexity(test_sent_new[i])
    score[i,2] = model_spa.score(test_sent_new[i])
    perplexity[i,2] = model_spa.perplexity(test_sent_new[i])
    score[i,3] = model_fra.score(test_sent_new[i])
    perplexity[i,3] = model_fra.perplexity(test_sent_new[i])
    score[i,4] = model_por.score(test_sent_new[i])
    perplexity[i,4] = model_por.perplexity(test_sent_new[i])
    score[i,5] = model_ita.score(test_sent_new[i])
    perplexity[i,5] = model_ita.perplexity(test_sent_new[i])
    
    
y_true = np.array(test_lang.map({'eng':0,'deu':1,'spa':2,'fra':3,'por':4,'ita':5}))
y_pred = np.argmax(score,axis=1)
Test_result = np.array([y_true,y_pred]).T
Test_result = pd.DataFrame(Test_result,columns=['y_true,y_pred'])
Test_result.columns('y_true','y_pred')


#y_pred = np.argmin(perplexity,axis=1)

#%% Step 5: Calculate metrics

import sklearn.metrics
import seaborn as sns

# overall accuracy
accuracy = sklearn.metrics.accuracy_score(y_true,y_pred)

# confusion matrix  - row: true | column: prediction
print(sklearn.metrics.confusion_matrix(y_true,y_pred))

# classification report: difference between accuracy & precision
print(sklearn.metrics.classification_report(y_true,y_pred,digits=4))

# Confusion Matrix Heatmap
lang = ['eng','deu','spa','fra','por','ita']
conf_matrix = sklearn.metrics.confusion_matrix(y_true,y_pred)
test_data = pd.DataFrame(conf_matrix,columns=lang,index=lang)
test_data

plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
sns.set(font_scale=1.4)
ax = plt.axes()
sns.heatmap(test_data,cmap='coolwarm',ax=ax,annot=True,fmt='.5g')
ax.set_title('Figure 2: N-gram Model Confusion Matrix Heatmap')
plt.show()



