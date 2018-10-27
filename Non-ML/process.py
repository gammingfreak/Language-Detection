#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:53:19 2018

"""

import sys
import nltk

tokenizor = nltk.tokenize.RegexpTokenizer("[a-zA-Z'`éèî]+") # The re expression needs to be edited

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(tokenizor.tokenize(sentence)).lower())
