# Copyright (c) Usman Sadiq
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
import pickle

import matplotlib
import matplotlib.pyplot as plt


def split_data(site, train_perc):
    """ Split data into training and test indices """
    train_indices = []
    test_indices = []

    for s in np.unique(site):
        # Make sure each site is represented in both training and test sets
        id_in_site = np.argwhere(site == s).flatten()

        num_nodes = len(id_in_site)
        train_num = int(train_perc * num_nodes)

        prng.shuffle(id_in_site)
        train_indices.extend(id_in_site[:train_num])
        test_indices.extend(id_in_site[train_num:])

    # print("Number of labeled samples %d" % len(train_indices))

    return train_indices, test_indices

rs = 33
print("Random state is %d" % rs)
prng = np.random.RandomState(rs)

def prepare_pairs(X, y, site, indices):
    """ Prepare the graph pairs before feeding them to the network """
    N, M, F = X.shape
    n_pairs = int(len(indices) * (len(indices) - 1) / 2)
    triu_pairs = np.triu_indices(len(indices), 1)

    X_pairs = np.ones((n_pairs, M, F, 2))
    X_pairs[:, :, :, 0] = X[indices][triu_pairs[0]]
    X_pairs[:, :, :, 1] = X[indices][triu_pairs[1]]

    site_pairs = np.ones(int(n_pairs))
    site_pairs[site[indices][triu_pairs[0]] != site[indices][triu_pairs[1]]] = 0

    y_pairs = np.ones(int(n_pairs))
    y_pairs[y[indices][triu_pairs[0]] != y[indices][triu_pairs[1]]] = 0  # -1

    print(n_pairs)

    return X_pairs, y_pairs, site_pairs

def ABIDE_save(num_subjects, filename):
    rs = 33
    print("Random state is %d" % rs)
    prng = np.random.RandomState(rs)

    # Split into training, validation and testing sets
    training_num = num_subjects;
    lines = int(1.2 * num_subjects);
    # I am guessing this is to create a validation set within training data
    # Used in the next moving step

    # Get subject features
    atlas = 'ho'
    kind = 'correlation'

    subject_IDs = abide_utils.get_ids(lines)
    # Get all subject networks
    networks = abide_utils.load_all_networks(subject_IDs, kind, atlas_name=atlas)
    X = np.array(networks)

    # with open('GCN_train.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(X, f, 2)
    # f.close()

    # Number of nodes
    nodes = X.shape[1]

    # Get ROI coordinates
    coords = abide_utils.get_atlas_coords(atlas_name=atlas)

    # Get subject labels
    label_dict = abide_utils.get_subject_label(subject_IDs, label_name='DX_GROUP')
    y = np.array([int(label_dict[x]) - 1 for x in sorted(label_dict)])

    # Get site ID
    site = abide_utils.get_subject_label(subject_IDs, label_name='SITE_ID')
    unq = np.unique(list(site.values())).tolist()
    site = np.array([unq.index(site[x]) for x in sorted(site)])

    # Choose site IDs to include in the analysis
    site_mask = range(20)
    X = X[np.in1d(site, site_mask)]
    y = y[np.in1d(site, site_mask)]
    site = site[np.in1d(site, site_mask)]

    tr_idx, test_idx = split_data(site, 0.6)
    # training_num = int(0.6 * X.shape[0])

    prng.shuffle(test_idx)
    subs_to_add = training_num - len(tr_idx)  # subjects that need to be moved from testing to training set
    tr_idx.extend(test_idx[:subs_to_add])
    test_idx = test_idx[subs_to_add:]
    print("The test indices are the following: ")
    print(test_idx)

    all_combs = []
    tr_mat = np.array(tr_idx).reshape([int(len(tr_idx) / 6), 6])
    for i in range(3):
        x1 = tr_mat[:, i * 2].flatten()
        x2 = tr_mat[:, i * 2 + 1].flatten()
        combs = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
        all_combs.append(combs)

    all_combs = np.vstack(all_combs)

    # print(all_combs.shape)
    n, m, f = X.shape
    X_train = np.ones((all_combs.shape[0], m, f, 2), dtype=np.float32)
    y_train = np.ones(all_combs.shape[0], dtype=np.int32)
    site_train = np.ones(all_combs.shape[0], dtype=np.int32)

    for i in range(all_combs.shape[0]):
        X_train[i, :, :, 0] = X[all_combs[i, 0], :, :]
        X_train[i, :, :, 1] = X[all_combs[i, 1], :, :]
        if y[all_combs[i, 0]] != y[all_combs[i, 1]]:
            y_train[i] = 0  # -1
        if site[all_combs[i, 0]] != site[all_combs[i, 1]]:
            site_train[i] = 0

    print("Training samples shape")
    print(X_train.shape)

    # Get the graph structure
    dist, idx = graph.distance_scipy_spatial(coords, k=10, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)

    graphs = []
    for i in range(3):
        graphs.append(A)

    # Calculate Laplacians
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    # Number of nodes in graph and features
    print("Number of controls in the dataset: ")
    print(y.sum())

    # Prepare training testing and validation sets
    X_test, y_test, site_test = prepare_pairs(X, y, site, test_idx)

    # Saving training data for comparison with ann siamese

    np.savez(filename,'.npz', name1=X_train, name2=y_train, name3=X_test, name4=y_test, name5=all_combs,
             name6=site_train, name7=site_test, name8=tr_idx)


    return None
