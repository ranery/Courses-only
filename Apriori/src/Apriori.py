# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from collections import defaultdict
from itertools import chain, combinations

class Apriori():
    def name(self):
        return 'Apriori Implementation'

    def getItemTransaction(self, data):
        """ return itemSet and transactionList from the data """
        self.itemSet = set()
        self.transactionList = list()
        for record in data:
            transaction = frozenset(record)
            self.transactionList.append(transaction)
            for item in transaction:
                self.itemSet.add(frozenset([item]))

    def ItemWithMinSupport(self, itemSet, transactionList, minSupport, freqSet):
        """ return items belong to itemSet which support value is not less than minSupport """
        localSet = defaultdict(int)
        for item in itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    freqSet[item] += 1
                    localSet[item] += 1
        newItemSet = set()
        for item, count in localSet.items():
            support = float(count) / len(transactionList)
            if support >= minSupport:
                newItemSet.add(item)
        return newItemSet

    def joinSet(self, itemSet, length):
        """ return joint set which length fixed """
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

    def subSet(self, var):
        """ return non-empty subsets of var """
        return chain(*[combinations(var, i+1) for i, a in enumerate(var)])

    def run(self, data, minSupport, minConfidence):
        """
        :param data: record iterator
        :param minSupport: the minimum support
        :param minConfidence: the minimum confidence
        :return: - items - rules
        """
        # get items set and transaction list
        self.getItemTransaction(data)
        # freqSet is the whole counter of all possible items
        freqSet = defaultdict(int)
        L = dict()
        associateRules = dict()
        current_C = self.ItemWithMinSupport(self.itemSet, self.transactionList, minSupport, freqSet)
        current_L = current_C
        k = 2
        while(current_L != set([])):
            L[k-1] = current_L
            current_L = self.joinSet(current_L, k)
            current_C = self.ItemWithMinSupport(current_L, self.transactionList, minSupport, freqSet)
            current_L = current_C
            k += 1

        # get items that satisfy the condition that more than minimum support value set by user
        items = []
        for key, value in L.items():
            _item = []
            for item in value:
                support = float(freqSet[item]) / len(self.transactionList)
                _item.append((tuple(item), support))
            items.extend(_item)

        # get rules that satisfy the condition that more than the minimum confidence value set by user
        rules = []
        for key, value in L.items():
            for item in value:
                support_item = float(freqSet[item]) / len(self.transactionList)
                _subsets = map(frozenset, [example for example in self.subSet(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:    # exclude item-self
                        support_element = float(freqSet[element]) / len(self.transactionList)
                        confidence = support_item / support_element
                        if confidence >= minConfidence:
                            rules.append(((tuple(element), tuple(remain)), confidence))  # x --> (l-x)
        return items, rules

    def printResults(self, items, rules):
        print('-----------Items------------')
        for item, support in sorted(items):
            print('item: %s, %.3f' % (str(item), support))
        print('-----------Rules------------')
        for rule, confidence in sorted(rules):
            A, B = rule
            print('Rule: %s --> %s, %.3f' % (str(A), str(B), confidence))