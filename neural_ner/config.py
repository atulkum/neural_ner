import os

class Config(object):
    pass

config = Config()
root_dir = os.path.join(os.path.expanduser('~'), 'Downloads/sequence_prediction-master')
config.data_dir = os.path.join(root_dir, 'CoNLL-2003')
config.log_root = os.path.join(root_dir, 'log')

config.train_file='eng.train'
config.validation_file='eng.testa'
config.test_file='eng.testb'
config.label_type= 'iobes' #'iob'

config.lr = 0.001
config.dropout_ratio = 0.5

config.max_grad_norm = 5.0
config.batch_size = 32
config.num_epochs = 100

config.print_every = 100

config.reg_lambda = 0.00007

config.dropout_rate = 0.5

config.lower = True
config.zeros = True
config.random_init = True

config.char_embd_dim = 25
config.char_lstm_dim = 25
config.word_emdb_dim = 100
config.word_lstm_dim = 100
config.caps_embd_dim = 3

config.glove_path = os.path.join(root_dir, 'neural_ner/.vector_cache/glove.6B.100d.txt')

config.vocab_size = int(4e5)

config.is_cuda = False

config.is_l2_loss = True