# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import h5py
import numpy as np
import tqdm

h5_path = 'HG002-Trust-ALL-QScores-Strand-multiAF.hdf'
hdfile = h5py.File(h5_path, 'r')
record = hdfile['data']
print(record.shape)
sample_h5_path = 'HG002-Trust-ALL-QScores-Strand-multiAF-8th.hdf'
sample_hf = h5py.File(sample_h5_path, 'w')
perm = np.random.permutation(range(record.shape[0]))
print(perm[0:10])
sort_perm = np.sort(perm[0:int(record.shape[0]/8.)])
print(sort_perm.shape)
print(sort_perm[0:10])
print(sort_perm[-10:])

batch_size = 10000
print(batch_size)
init_dataset = record[list(sort_perm[0:batch_size])]
df_sample = sample_hf.create_dataset("data", maxshape=(None,), data=init_dataset, compression="gzip")
print(df_sample.shape)
print(df_sample.dtype)
batches = sort_perm.shape[0] / batch_size
print(batches)
batches = int(batches) - 2
for b in tqdm.tqdm(range(1,batches+1), total=batches):
    print('%s: %s' % (b, sort_perm[batch_size*b:batch_size*(b+1)].shape))
    d_len = df_sample.shape[0]
    df_sample.resize((d_len + batch_size,))
    df_sample[d_len:] = record[list(sort_perm[batch_size*b:batch_size*(b+1)])]
