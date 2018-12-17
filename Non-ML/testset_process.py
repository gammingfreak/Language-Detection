#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 09:43:35 2018

@author: Cathy
"""

# Test file


import sys
import nltk

#tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`àéèî]+") # The re expression needs to be edited
tokenizor = nltk.tokenize.RegexpTokenizer("['a-zA-ZÀ-ÖØ-öø-ÿ]+")

temp = []
token = []
    
for line in sys.stdin:
    token.append(' '.join(tokenizor.tokenize(line)).lower())

for line in token:
    sys.stdout.write(line)
    sys.stdout.write('\n')

'''
for line in temp:
    for sentence in nltk.sent_tokenize(line):
        token.append(' '.join(tokenizor.tokenize(sentence)).lower())

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(nltk.word_tokenize(sentence)).lower())
'''
