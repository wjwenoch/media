
use_cp_tag_only = True


spacy_tag_dep = dict(
    tag_dep_seq_max_len = 26, # must be >= max question length!
    tag_dim = 200,
    lstm_units= 64,
    batch_size= 100, # default value for None is 32
    epochs=200,
    verbose=2,
    num_dense_dropout = 1,
    dropout_rate = 0.5,
    output_num_tags = 3 if use_cp_tag_only else 5, # i.e., ['O', 'BC', 'BP', 'IC', 'IP']
    model_save_name = 'last_epoch_model.h5',
)

embedding_dataset_list=dict(
    glove='glove',
    conceptnet='conceptnet',
)



logging_format = '[%(asctime)s] %(module)s(%(levelname)s)- %(message)s'

import logging
logging.basicConfig(format=logging_format, datefmt='%Y-%m-%d, %H:%M:%S', level=logging.INFO)