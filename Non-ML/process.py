#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:53:19 2018

<<<<<<< HEAD
=======
@author: Yifang
>>>>>>> master
"""

import sys
import nltk

<<<<<<< HEAD
tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`éèî]+") # The re expression needs to be edited
=======

#tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`éèî]+") # The re expression needs to be edited
tokenizor = nltk.tokenize.RegexpTokenizer("['a-zA-ZÀ-ÖØ-öø-ÿ]+")
>>>>>>> master

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(tokenizor.tokenize(sentence)).lower())
<<<<<<< HEAD
=======

'''
for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(nltk.word_tokenize(sentence)).lower())
'''
>>>>>>> master
