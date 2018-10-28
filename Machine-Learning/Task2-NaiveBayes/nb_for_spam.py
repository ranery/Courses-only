# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class naive_bayes():
    def name(self):
        return 'naive bayes classifier'

    def train(self, dataset, classes, m='bernouli'):
        """
        :param dataset: all doc_vectors
        :param classes: spam or not
        :param m : m-esitmation methods
        condition_prob : conditional probability p(w|s)
        cls_prob : prior probability p(s)
        """
        sub_dataset = defaultdict(list)
        cls_cnt = defaultdict(lambda :0)
        for doc_vector, cls in zip(dataset, classes):
            sub_dataset[cls].append(doc_vector)
            cls_cnt[cls] += 1
        self.cls_prob = {k: v/len(classes) for k, v in cls_cnt.items()}
        self.condition_prob = {}
        dataset = np.array(dataset)
        for cls, sub_dataset in sub_dataset.items():
            # m-estimation
            sub_dataset = np.array(sub_dataset)
            if m == 'bernouli':
                self.condition_prob[cls] = np.log((np.sum(sub_dataset, axis=0) + 1) / (np.sum(dataset, axis=0) + 2))
            elif m == 'polynomial':
                self.condition_prob[cls] = np.log((np.sum(sub_dataset, axis=0) + 1) / (np.sum(dataset, axis=0) + len(sub_dataset[0])))
            else:
                self.condition_prob[cls] = np.log(np.sum(sub_dataset, axis=0) / np.sum(dataset, axis=0))

    def classify(self, doc_vector):
        posterior = {}
        for cls, cls_prob in self.cls_prob.items():
            condition_prob_vec = self.condition_prob[cls]
            posterior[cls] = np.sum(condition_prob_vec * doc_vector) + np.log(cls_prob)
        return max(posterior, key=posterior.get)

    def test(self, dataset, classes):
        error = 0
        for doc_vector, cls in zip(dataset, classes):
            pred = self.classify(doc_vector)
            print('Predict: {} --- Actual: {}'.format(pred, cls))
            if pred != cls:
                error += 1
        print('Error rate: {}'.format(error/len(classes)))

    def predict(self, dataset):
        if os.path.exists('results.csv'):
            os.remove('results.csv')
        f = open('results.csv', 'a', newline='')
        csv_write = csv.writer(f, dialect='excel')
        i = 0
        for doc_vector in dataset:
            result = []
            i += 1
            pred = self.classify(doc_vector)
            result.append(i)
            result.append(pred)
            csv_write.writerow(result)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cls, prob in self.condition_prob.items():
            ax.scatter(np.arange(0, len(prob)),
                       prob*self.cls_prob[cls],
                       label=cls,
                       alpha=0.3)
            ax.legend()
        plt.show()
        plt.savefig

