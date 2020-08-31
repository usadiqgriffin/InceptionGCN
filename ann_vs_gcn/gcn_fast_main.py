# Copyright (c) 2017 Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from lib import models_siamese, graph, abide_utils
import numpy as np
import os
import time

# Get subject features
atlas = 'ho'
kind = 'correlation'

#
# Get ROI coordinates
coords = abide_utils.get_atlas_coords(atlas_name=atlas)

# Load data first

def gcn_run(fname,train_num,batch_size):
    rs = 222

    print("Random state is %d" % rs)
    prng = np.random.RandomState(rs)

    np.random.seed(seed=222)
    data = np.load(fname+'.npz');

    X_train=np.array(data['name1'],dtype=np.float32);
    y_train=np.array(data['name2'],dtype=np.float32);

    X_test=np.array(data['name3'],dtype=np.float32);
    y_test=np.array(data['name4'],dtype=np.float32);

    all_combs=np.array(data['name5'],dtype=np.float32);
    site_train=np.array(data['name6'],dtype=np.float32);
    site_test=np.array(data['name7'],dtype=np.float32);
    tr_idx=np.array(data['name8'],dtype=np.float32);

    dist, idx = graph.distance_scipy_spatial(coords, k=10, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)

    graphs = []
    for i in range(3):
        graphs.append(A)

    # Calculate Laplacians
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    n, m, f, _ = X_train.shape

    # Graph Conv-net
    features = 64
    K = 3
    params = dict()
    params['num_epochs'] = train_num
    params['batch_size'] = batch_size
    # params['eval_frequency'] = X_train.shape[0] / (params['batch_size'] * 2)
    params['eval_frequency'] = 1
    # Building blocks.
    params['filter'] = 'chebyshev5'
    params['brelu'] = 'b2relu'
    params['pool'] = 'apool1'

    # Architecture.
    params['F'] = [features, features]  # Number of graph convolutional filters.
    params['K'] = [K, K]  # Polynomial orders.
    params['p'] = [1, 1]  # Pooling sizes.
    params['M'] = [1]  # Output dimensionality of fully cofeannected layers.
    params['input_features'] = f
    params['lamda'] = 0.35
    params['mu'] = 0.6

    # Optimization.
    params['regularization'] = 5e-3
    params['dropout'] = 0.8
    params['learning_rate'] = 1e-2
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = X_train.shape[0] / params['batch_size']

    params['dir_name'] = 'siamese_' + time.strftime("%Y_%m_%d_%H_%M") + '_feat' + str(params['F'][0]) + '_' + \
                         str(params['F'][1]) + '_K' + str(K) + '_state'

    print(params)

    # Run model
    model = models_siamese.siamese_cgcnn_cor(L, **params)

    print("Constructor finished")
    accuracy, loss, scores_summary, tr_error, test_error = model.fit(X_train, y_train, site_train, X_test, y_test, site_test)
    #print('Time per step: {:.2f} ms'.format(t_step*1000))

    # Save training
    tr_res = model.evaluate(X_train, y_train, site_train)

    # Evaluate test data
    print("Test accuracy is:")
    res = model.evaluate(X_test, y_test, site_test)
    print(res[0])


    return tr_error,test_error



