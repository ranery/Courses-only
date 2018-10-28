# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import itertools
import random

def get_doc_vector(words, vocabulary):
    doc_vector = [0] * len(vocabulary)
    for word in words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            doc_vector[idx] = 1
    return doc_vector

def parse_file(dir, vocabulary, word_vector, classes, has_cls=True):
    dir_list = os.listdir(dir)
    dir_list.sort(key=lambda x:int(x[:-4]))
    for i in range(0, len(dir_list)):
        path = os.path.join(dir, dir_list[i])
        if os.path.isfile(path):
            words = []
            with open(path, 'r', encoding='ISO-8859-1') as f:
                for line in f:
                    if line:
                        vocabulary.extend(line.strip())
                        words.append(line.strip())
                        words.append(' ')
            if has_cls: classes.append(dir[13:-1])
            word_vector.append(''.join(itertools.chain(words)))
    vocabulary = list(set(vocabulary))
    if has_cls:
        return vocabulary, word_vector, classes
    else:
        return vocabulary, word_vector

def split_val(dataset, cls):
    for i in range(0, len(dataset)):
        dataset[i].append(cls[i])
    train = random.sample(dataset, int(0.8*len(dataset)))
    val = random.sample(dataset, len(dataset)-int(0.8*len(dataset)))
    # val = [example for example in dataset if example not in train]
    train_dataset, train_cls, val_dataset, val_cls = [], [], [], []
    for i in range(0, len(train)):
        train_cls.append(train[i][-1])
        train_dataset.append(train[i][:-1])
    for i in range(0, len(val)):
        val_cls.append(val[i][-1])
        val_dataset.append(val[i][:-1])
    return train_dataset, train_cls, val_dataset, val_cls

def dataset():
    vocabulary, train_word_vector, train_cls = parse_file('./train_data/ham/', [], [], [])
    vocabulary, train_word_vector, train_cls = parse_file('./train_data/spam/', vocabulary, train_word_vector, train_cls)
    vocabulary, test_word_vector = parse_file('./test_data/', vocabulary, [], [], False)
    train_dataset = [get_doc_vector(words, vocabulary) for words in train_word_vector]
    test_dataset = [get_doc_vector(words, vocabulary) for words in test_word_vector]
    train_dataset, train_cls, val_dataset, val_cls = split_val(train_dataset, train_cls)
    print('num of trainset : ', len(train_dataset))
    print('num of valset   : ', len(val_dataset))
    print('num of testset  : ', len(test_dataset))
    return train_dataset, train_cls, val_dataset, val_cls, test_dataset