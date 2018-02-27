import json
import logging
import numpy as np
import os

from  ..python import config
#from .load_data import check_longest_sequence, media_vec_file

'''
train dataset size: 12908
val dataset size: 1259
test dataset size: 3005
total dataset voc size: 2427
total label size: 138
train max length is 192
val max length is 85
test max length is 158
'''

media_train = 'D:/datasets/atis/media/train.crf'
media_val = 'D:/datasets/atis/media/dev.crf'
media_test = 'D:/datasets/atis/media/test.crf'
media_vec_file = 'D:/datasets/atis/media/fasttext_voc.json'


def check_longest_sequence(data):
    for d in data:
        xlen = max(list(map(len, d)))
        logging.info("max length is %s", xlen)
    return


def load_media_data(fp):
    data = dict()
    qidpref= 'Q'
    count = 1
    with open(fp, 'r', encoding='latin1') as f:
        sent = []
        ent = []
        lb = []
        for r in f:
            r = r.strip()
            if len(r) <= 0:
                if len(sent) > 0:
                    data[qidpref+str(count)] = {
                        'text':sent.copy(),
                        'label':lb.copy(),
                        'entity':ent.copy()
                    }
                    sent[:] = []
                    lb[:] = []
                    ent[:] = []
                    count += 1
                continue
            items = r.split()
            sent.append(items[0])
            ent.append(items[1])
            lb.append(items[2])
    with open(fp+'.json', 'w', encoding='utf8') as f:
        json.dump(data, f)


french_fasttext_fp = 'D:/datasets/embeddings/fasttext.wiki.fr.vec'
def get_fasttext_media(voc):
    if os.path.exists(media_vec_file):
        return
    mat = dict()
    with open(french_fasttext_fp, 'r', encoding='utf8' ) as f:
        for line in f:
            values = line.split(None)
            word = values[0]
            if word not in voc:
                continue
            try:
                float(values[1])
            except ValueError:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            mat[word] = coefs.tolist()
    # fix the missing ones
    exist = set(mat.keys())
    diff = set(voc).difference(exist)
    logging.warning("the following terms are not found in pretrained vecs: ", list(diff))
    with open(media_vec_file, 'w' ) as f:
        json.dump(mat, f)


import collections
def prepare_data():
    def get_components(fp):
        with open(fp+'.json', 'r' ) as f:
            data = json.load(f)
            #sorted_data = collections.OrderedDict(data)
            x = []
            y = []
            ents = []
            for k,v in data.items():
                x.append(v['text'])
                y.append(v['label'])
                ents.append(v['entity'])
        logging.info("dataset size: %s", len(x))
        return x, y, ents
    train_x, train_label, train_ents = get_components(media_train)
    val_x,  val_label, val_ents  = get_components(media_val)
    test_x, test_y, test_ents = get_components(media_test)
    voc = set()
    voc.update([item for sublist in train_x for item in sublist])
    voc.update([item for sublist in val_x for item in sublist])
    voc.update([item for sublist in test_x for item in sublist])
    voc = sorted(list(voc))
    logging.info("media dataset voc size: %s", len(voc))
    get_fasttext_media(voc)
    idx2w = dict((i,w) for i,w in enumerate(voc))
    voc_lb = set()
    voc_lb.update([item for sublist in train_label for item in sublist])
    voc_lb.update([item for sublist in val_label for item in sublist])
    voc_lb.update([item for sublist in test_y for item in sublist])
    voc_lb = sorted(list(voc_lb))
    logging.info("media dataset label size: %s", len(voc_lb))
    idx2la = dict((i,w) for i,w in enumerate(voc_lb))
    test_len = False
    if test_len:
        logging.info("checking longest in x")
        check_longest_sequence([train_x, val_x, test_x])
        logging.info("checking longest in y labels")
        check_longest_sequence([train_label, val_label, test_y])
    w2idx = dict((w,i) for i,w in idx2w.items())
    lb2idx = dict((w,i) for i,w in idx2la.items())
    def word2index(data, indexer):
        newdata = []
        for d in data:
            newdata.append([indexer[item] for item in d])
        return newdata
    train_x = word2index(train_x, w2idx)
    val_x = word2index(val_x, w2idx)
    test_x = word2index(test_x, w2idx)
    train_label = word2index(train_label, lb2idx)
    val_label = word2index(val_label, lb2idx)
    test_y = word2index(test_y, lb2idx)
    res = {}
    res['train_x'] = train_x
    res['train_y'] = train_label
    res['val_x'] = val_x
    res['val_y'] = val_label
    res['test_x'] = test_x
    res['test_y'] = test_y
    res['idx2w'] = idx2w
    res['idx2lb'] = idx2la
    res['vocab_size'] = len(voc)
    res['train_ents'] = train_ents
    res['val_ents'] = val_ents
    res['test_ents'] = test_ents
    all_ents = train_ents+val_ents+test_ents
    all_ents = [i for l in all_ents for i in l]
    all_ents = set(all_ents)
    logging.info("total %s of ents in MEDIA", len((all_ents)))
    return res



media_json_fp = 'files/media_entity_feature.json'
def load_media_entitynames(recompute = False, use_all_entities = True, binary_matrix = True):
    def get_w2e(fp):
        w2e = dict()
        with open(fp, 'r', encoding='latin1') as f:
            for r in f:
                r = r.strip()
                if len(r) <= 0:
                    continue
                items = r.split()
                w = items[0]
                e = items[1]
                if not use_all_entities and w == e:
                    continue
                w2e[w] = e
        return w2e
    if not recompute and os.path.exists(media_json_fp):
        with open(media_json_fp, 'r') as f:
            w2e = json.load(f)
    else:
        e1 = get_w2e(media_train)
        e2 = get_w2e(media_val)
        e3 = {**e1, **e2}
        e4 = get_w2e(media_test)
        e3 = {**e3, **e4}
        with open(media_json_fp, 'w') as f:
            json.dump(e3, f)
        w2e = e3
    enm = None
    entmap = None
    if binary_matrix:
        fs = list(set(w2e.values()))
        enm = np.zeros((len(fs) + 1, len(fs)))
        for  i in range(len(fs)):
            enm[i+1][i] = 1
        #update the value index in w2e, so index are directly given for embeddings
        entmap = dict((f, i+1) for i, f in enumerate(fs))
    return [enm, entmap, w2e]