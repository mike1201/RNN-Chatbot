import tensorflow as tf
import numpy as np
import sys

import seq2seq
import data_utils


metadata, idx_q, idx_a = data_utils.load_data(PATH='datasets/')
# print(idx_q.shape, idx_a.shape)
# print(idx_q[0])

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)


xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
id2word = metadata['idx2w'] 
xvocab_size = len(id2word) 
yvocab_size = xvocab_size
emb_dim = 300


train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)


model = seq2seq.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               id2word=id2word,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=2
                               )


sess = model.train(train_batch_gen, val_batch_gen)
output = model.predict(sess, test_batch_gen, 16)
