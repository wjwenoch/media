from ..python import config,data_gen
from .media import load_media_entitynames, media_vec_file
import logging, os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate, multiply,  Input, \
    Lambda, Permute,  TimeDistributed, Embedding
from keras.layers.core import Flatten, Dropout,  Masking, Dense
from keras.models import optimizers, Model
from keras import utils,regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import datetime as dt
from keras.callbacks import CSVLogger


data_dir = 'D:/datasets/atis'


USE_MEDIA_DATA = True # no pretrained embedding so far
MEDIA_ENTITY_ONLY = True

now = dt.datetime.now()
atis_max_qlen = 158 if USE_MEDIA_DATA else 46
num_output_classes = 138 if USE_MEDIA_DATA else 127 # 127 incl. O

MEDIA_ENTITY_VOCAB_SIZE = 1671
ENTITY_TOTAL_NUM = MEDIA_ENTITY_VOCAB_SIZE if USE_MEDIA_DATA else 37 # not to be used when context window is not used


model_save_dir = 'models/'

USE_CONTEXT_WINDOW = True
USE_CONTEXT_ENTITY = True
CONTEXT_WINDOW_PADSIZE = 3 if USE_MEDIA_DATA else 5 ## this is one-sided value

# entity can be used as embedding or one-hot encoding, try learning embedding first
USE_ENTITY_AS_SEQUENCE = not USE_CONTEXT_ENTITY and True


CHECK_MAX_PADDING = False # train: 42, val: 46, test: 30


hyper_params = dict(
    opt_nadam= False,
    lstm_input_dropout = 0.0,
    lstm_state_dropout = 0.0,
    two_bilstm = False,
)



def get_entity_sequence(data, idx2w, w2e, entmap):
    entity = []
    idxw2e = dict((wi, entmap[w2e[idx2w[wi]]]) for wi in list(idx2w.keys()))
    for d in data:
        d = list(d)
        nd = [idxw2e[w] for w in d]
        entity.append(nd)
    return entity


#this method is designed for  entities, e.g., input must be batch X timeSteps X ?
# the output will be batch X timeSteps X ? X entities_one_hot_size (37 for now)
def get_entity_index_onehot(x, idx2w, enm, entmap, w2e, concatenated = True):
    ret = []
    for e in x:
        if e < 0:
            xp = enm[0]
        else:
            w = idx2w[e]
            ent_idx = 0 if w not in w2e else entmap[w2e[w]]
            xp = enm[ent_idx]
        if concatenated:
            ret.extend(xp)
        else:
            ret.append(xp)
    return ret


# note that entmap is indexed from 1, not 0.
def get_entity_index(x, idx2w, w2e, entmap):
    wi2ei = dict((wi, entmap[w2e[idx2w[wi]]]-1) for wi in list(idx2w.keys()))
    ret = []
    for e in x:
        if e < 0:
            xp = -1
        else:
            xp = wi2ei[e]
        ret.append(xp)
    return ret


def context_window(data, idx2w, enm, entmap, w2e, special_idx = -1):
    new_data = []
    new_ents = []
    for l in data:
        l = list(l)
        lpadded = CONTEXT_WINDOW_PADSIZE * [special_idx] + l + CONTEXT_WINDOW_PADSIZE * [special_idx]
        out = [lpadded[i:(i + 2*CONTEXT_WINDOW_PADSIZE +1)] for i in range(len(l))]
        if USE_CONTEXT_ENTITY:
            #ent_out = [get_entity_index_onehot(x, idx2w, enm, entmap, w2e) for x in out]
            ent_out = [get_entity_index(x, idx2w, w2e, entmap) for x in out]
            new_ents.append(ent_out)
            if max(ent_out[0]) > 1670:
                logging.info("max idx %s", max(ent_out[0]))
        new_data.append(out)
    return new_data, new_ents






def check_longest_sequence(data):
    for d in data:
        xlen = max(list(map(len, d)))
        logging.info("max length is %s", xlen)
    return


def convert2data(x, y, idx2w, idx2la, padding_len = atis_max_qlen,
                 upto=float('inf')):
    enm = None
    entmap = None
    w2e = None
    if USE_CONTEXT_ENTITY or USE_CONTEXT_WINDOW or USE_ENTITY_AS_SEQUENCE:
        if USE_MEDIA_DATA:
            enm, entmap, w2e =  load_media_entitynames(use_all_entities=True, recompute=True)
    encs = []
    entities = []
    encs_lbs = []
    wts = []
    pads = []
    glove = None
    cate = None
    count = 0
    for i, q in enumerate(x):
        if count > upto:
            break
        count += 1
        if USE_CONTEXT_WINDOW:
            cont, ents = context_window([q], idx2w, enm, entmap, w2e)
            q = cont[0] if cont else q
            ents = ents[0] if ents else None
        elif USE_ENTITY_AS_SEQUENCE:
            # add entities here
            ents = get_entity_sequence([q], idx2w, w2e, entmap)[0]
        #if USE_PRETRAINED_EMBEDDING:
        q = np.array(q) + 1
        q = q.tolist()
        # entities are used as is, not one-hot
        ents = np.array(ents) + 1
        ents = ents.tolist()
        #
        kb = None
        kb_distance_dims = None
        actual_len = len(q)
        wt = np.ones(len(q))
        seq = q
        seq = pad_sequences([q], maxlen=padding_len, padding='post', truncating='post')[0]
        if USE_CONTEXT_ENTITY or USE_ENTITY_AS_SEQUENCE:
            ents = pad_sequences([ents], maxlen=padding_len, padding='post', truncating='post')[0]
            entities.append(ents)
        encs.append(seq)
        lb = y[i]
        #seq_y =  pad_sequences([lb], maxlen=padding_len, padding='post', truncating='post')[0]
        cat = utils.to_categorical([lb], num_classes=num_output_classes)
        diff = padding_len - actual_len
        pads.append(diff)
        if diff > 0:
            wts.append( np.concatenate( (np.array(wt), np.full(diff, 0)) ))
            pad_item = np.zeros( (diff, num_output_classes) )
            cat = np.concatenate( (cat, pad_item) )
            encs_lbs.append(cat)
        elif diff<0:
            logging.warning("padding %s is too short for this input of length %s", atis_max_qlen, cat.shape[0])
            wt = wt[0:atis_max_qlen]
            wts.append(wt)
            cat = cat[0:atis_max_qlen]
            encs_lbs.append(cat)
        else:
            wts.append(wt)
            encs_lbs.append(cat)
    if USE_CONTEXT_WINDOW:
        encs = np.array(encs)
        encs = np.transpose(encs, axes=(2,0,1))
        if USE_CONTEXT_ENTITY:
            entities = np.array(entities)
            entities = np.transpose(entities, axes=(2,0,1))
        else:
            entities = np.array(entities)
    else:
        encs = np.array(encs)
    return [encs, entities, np.array(wts), np.array(encs_lbs),  pads]



def get_word_embeddings(pretrained_vecs, qinput, word_voc_size,
                        maxlen = None, name='embed'):
    qb = None
    if pretrained_vecs is None:
        qb = Embedding(word_voc_size + 1, config.spacy_tag_dep['tag_dim'],
                       embeddings_initializer = 'RandomNormal',
                       input_length=maxlen,
                       trainable=True,
                       mask_zero=True,
                       name=name)
        logging.info("learnable embedding layer created")
    else:
        qb = Embedding(word_voc_size+1, config.spacy_tag_dep['tag_dim'],
                       input_length=maxlen,
                       trainable=False,
                       mask_zero=True,
                       weights=[pretrained_vecs],
                       name=name)
        logging.info("pretrained embedding layer created")
    if qinput is None:
        return qb
    else:
        return qb(qinput)



def context_window_embeddins(voc_size, max_len, name_prefix='sent'):
    defs = []
    all_ems = []
    ei = get_word_embeddings(None, None,
                                             word_voc_size=voc_size, maxlen=max_len,
                                             name=name_prefix+'_embed')
    for i in range(2*CONTEXT_WINDOW_PADSIZE+1):
        ipi = Input(shape=(max_len,), name=name_prefix + "input"+str(i))
        defs.append(ipi)
        all_ems.append(ei(ipi))
    return defs, all_ems


def keras_model_MLP_gen(trainx, trainy, valx, valy, voc_size, idx2w, idx2la, upto=float('inf')):
    if upto and upto > 0 :
        trainx = trainx[0:upto]
        trainy = trainy[0:upto]
    qs_input = None
    input_embed = None
    ent_input = None
    if USE_CONTEXT_WINDOW:
        if MEDIA_ENTITY_ONLY:
            qs_input = []
            input_real = []
        else:
            input_defs, input_real = context_window_embeddins(voc_size, atis_max_qlen, name_prefix='sent')
            qs_input = input_defs
        if USE_CONTEXT_ENTITY:
            ent_input, ents_real = context_window_embeddins(MEDIA_ENTITY_VOCAB_SIZE, atis_max_qlen, name_prefix='entity')
            input_embed = concatenate(input_real+ents_real)
        else:
            if input_real:
                input_embed = concatenate(input_real)
    elif USE_ENTITY_AS_SEQUENCE:
        all_qs_inputs = []
        all_ems = []
        qi = Input(shape=(atis_max_qlen,), name="sent_input")
        ei = Input(shape=(atis_max_qlen,), name="entity_input")
        all_qs_inputs.append(qi)
        all_qs_inputs.append(ei)
        qi_eb =  get_word_embeddings(None, None,
                                                    word_voc_size=voc_size, maxlen=atis_max_qlen,
                                                    name='sent_embed')(qi)
        ent_eb = get_word_embeddings(None, None,
                                                     word_voc_size=MEDIA_ENTITY_VOCAB_SIZE, maxlen=atis_max_qlen,
                                                     name='ent_embed')(ei)

        all_ems.append(qi_eb)
        all_ems.append(ent_eb)
        qs_input = all_qs_inputs
        input_embed = concatenate(all_ems)
    else:
        qs_input = Input(shape=(atis_max_qlen,), name="sent_input")
        input_embed = get_word_embeddings(None, qs_input,
                                                          word_voc_size=voc_size, maxlen=atis_max_qlen)
    concat = input_embed
    drop1 = Dropout(0.2, name='dropout_1')(concat)
    hidden = TimeDistributed(Dense(units=200, activation='relu', name='dense_2',
                                   kernel_regularizer=regularizers.l2(1e-4)
                                   ))(drop1)
    drop2 =  Dropout(config.spacy_tag_dep['dropout_rate'] , name='dropout_2')(hidden)
    dense3 = TimeDistributed(Dense(units=num_output_classes, name='dense_3', activation='softmax'))(drop2) # no activation here
    output = dense3
    last_dim = int(output.shape[2])
    lambda_layer = Lambda(lambda x:x, output_shape=(atis_max_qlen,last_dim))
    output_layer = lambda_layer(output)
    if USE_CONTEXT_WINDOW:
        multi_inputs_definition = qs_input
        if USE_CONTEXT_ENTITY:
            multi_inputs_definition.extend(ent_input)
    elif USE_ENTITY_AS_SEQUENCE:
        multi_inputs_definition = qs_input
    else:
        multi_inputs_definition = [qs_input]
    final_model = Model(inputs=multi_inputs_definition, outputs=output_layer)
    loss_func = 'categorical_crossentropy'
    met = ['categorical_accuracy']
    opt = optimizers.Adam(lr=5e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    final_model.compile(optimizer=opt,
                        loss = loss_func,
                        metrics= met,
                        #sample_weight_mode='temporal',
                        )
    logging.info(final_model.summary())
    #monitor_quant = 'val_' + met[0] # accuracy based, if loss to be used, change to + 'loss' or acc for accuracy
    monitor_quant = 'val_loss'
    mlp_models = model_save_dir+'MLP_' + now.strftime('%m_%d_%H') + '_{epoch:03d}_ACC_{val_custom_categorical_accuracy:.3f}.hdf5'
    checkpoint_val = ModelCheckpoint(mlp_models, monitor=monitor_quant, verbose=1, save_best_only=True, mode='auto')
    early_stop_val = EarlyStopping(monitor=monitor_quant, patience=3, mode='auto', min_delta=-5)
    csvlogger = CSVLogger(model_save_dir+'log.csv', append=False, separator=',')
    callbacks_list = [
        early_stop_val,
        checkpoint_val,
        csvlogger,
    ]
    batch_size = config.spacy_tag_dep['batch_size']
    gen_params = {
        'batch_size' : batch_size,
        'shuffle' : True,
        'idx2w' : idx2w,
        'idx2la': idx2la,
    }
    'the generator needs to produce weights IF sammple_weight=temporal is used!'
    train_gen = data_gen.DataFeeder(**gen_params).generate(trainx, trainy)
    val_gen = data_gen.DataFeeder(**gen_params).generate(valx, valy)
    hist = final_model.fit_generator(generator=train_gen,
                                     steps_per_epoch= 1+(len(trainx)//batch_size),
                           verbose=config.spacy_tag_dep['verbose'],
                           epochs=config.spacy_tag_dep['epochs'],
                           callbacks=callbacks_list,
                           validation_data= val_gen,
                            validation_steps = 1+(len(valx)//batch_size),
                           #sample_weight=train_sample_wts,
                           )
    logging.info("training done, model saved. Now plotting..")



