# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import tkinter
import data
from Apriori import Apriori

class AprioriGUI(object):
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Apriori')
        # set input box
        self.label_S = tkinter.Label(self.root, text='minimum Support value :', anchor='c', bg='lightgray')
        self.minS_input = tkinter.Entry(self.root, width=20, bg='aliceblue')
        self.label_C = tkinter.Label(self.root, text='minimum Confidence value :', anchor='c', bg='lightgray')
        self.minC_input = tkinter.Entry(self.root, width=20, bg='aliceblue')
        # display list
        self.display_info = tkinter.Listbox(self.root, width=50, bg='snow')
        # result botton
        self.result_botton = tkinter.Button(self.root, command=self.run, text='Run', bg='oldlace')

    def gui_arrang(self):
        # self.label_S.grid(row=0,column=0)
        # self.label_C.grid(row=1,column=0)
        # self.minS_input.grid(row=0,column=1)
        # self.minC_input.grid(row=1,column=1)
        # self.display_info.grid(columnspan=2)
        # self.result_botton.grid(columnspan=2)
        self.label_S.pack(fill=tkinter.X)
        self.minS_input.pack(fill=tkinter.X)
        self.label_C.pack(fill=tkinter.X)
        self.minC_input.pack(fill=tkinter.X)
        self.display_info.pack()
        self.result_botton.pack()

    def run(self):
        self.minC = float(self.minC_input.get())
        self.minS = float(self.minS_input.get())
        # delete all
        self.display_info.delete(0,tkinter.END)
        # get dataset
        inputFile = data.dataset('goods.csv')
        # apriori
        items, rules = Apriori.run(inputFile, self.minS, self.minC)
        self.display_info.insert(0,'----------Items-----------')
        line = 0
        for item, support in sorted(items):
            line += 1
            self.display_info.insert(line,'item: {}, {}'.format(str(item), str(support)))
        line += 1
        self.display_info.insert(line,'----------Rules-----------')
        for rule, confidence in sorted(rules):
            line += 1
            self.display_info.insert(line,'rule: {}, {}'.format(str(rule), str(confidence)))


if __name__ == '__main__':
    Apriori = Apriori()
    AprioriGUI = AprioriGUI()
    AprioriGUI.gui_arrang()
    tkinter.mainloop()