from ..python import config

import logging
import numpy as np

from ..python import load_data

class DataFeeder():
    'to resolve memory issues, use data generator to feed data into keras model, esp. for the MEDIA dataset'
    def __init__(self, batch_size=32, shuffle=True, idx2w=None, idx2la=None):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.idx2w = idx2w
        self.idx2la = idx2la


    def __get_order(self, array_idx):
        if self.shuffle:
            np.random.shuffle(array_idx)
        return array_idx


    def __gen_data(self, subx, suby):
        questions, ents, train_sample_wts, output_labels, train_kbs, train_pads \
            = load_data.convert2json(subx, suby, self.idx2w, self.idx2la)
        X = None
        Y = output_labels
        if load_data.USE_CONTEXT_WINDOW:
            if load_data.MEDIA_ENTITY_ONLY:
                X = []
            else:
                X = [questions[i] for i in range(2*load_data.CONTEXT_WINDOW_PADSIZE+1)]
            if load_data.USE_CONTEXT_ENTITY:
                #X.append(ents)
                X.extend(ents)
        elif load_data.USE_ENTITY_AS_SEQUENCE:
            X = [questions, ents]
        else:
            X = [questions]
        if load_data.USE_KB_INPUT:
            X.append(train_kbs)
        return X,Y


    '''
    infinite loop over the data
    '''
    def generate(self, data_x, data_y):
        while True:
            idx = list(range(len(data_x)))
            idx = self.__get_order(idx)
            #generate batches
            max_batch = len(data_x)//self.batch_size
            for i in range(max_batch+1):
                end = (i+1)*self.batch_size
                end = min([end, len(data_x)])
                items = [k for k in idx[i*self.batch_size:end]]
                subx = [data_x[k] for k in items]
                suby = [data_y[k] for k in items]
                x,y = self.__gen_data(subx, suby)
                yield x,y
