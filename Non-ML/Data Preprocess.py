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