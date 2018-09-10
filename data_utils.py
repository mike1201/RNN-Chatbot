import numpy as np
from random import sample
import pickle

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')

    return metadata, idx_q, idx_a


def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx], y[sample_idx]


def decode(sequence_batch, id2word, separator=' '): # 0 used for padding, is ignored
    decoded_batch = []
    for sequence in sequence_batch:
        decoded_element_list = []
        for element in sequence:
            decoded_element_list.append( id2word[element] )
        decoded_sequence = separator.join( decoded_element_list )
        decoded_batch.append(decoded_sequence)

    return decoded_batch
