# Software by Jeremy Kawahara and Colin J Brown
# Medical Image Analysis Lab, Simon Fraser University, Canada, 2017
# Simple "Hello World" example.

import os, sys
import numpy as np
import cPickle as pickle
from scipy.stats.stats import pearsonr

sys.path.append("/home/usmansadiq/caffe/python")
import caffe

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))  # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN

np.random.seed(seed=22)  # To reproduce results.

# Loading training data to compare:

data = np.load('/home/usmansadiq/DeepLearn/fMRI/gcn_metric_learning/conn_train_med.npz')

injury = ConnectomeInjury()  # Generate train/test synthetic data.
x_sim, y_sim = injury.generate_injury()

x_tr1=np.array(data['name1'],dtype=np.float32);
y_tr1=np.array(data['name2'],dtype=np.float32);

x_val=np.array(data['name3'],dtype=np.float32);
y_val=np.array(data['name4'],dtype=np.float32);

# Reshape data for ann

x_train=np.ndarray.swapaxes(x_tr1,1,3);
x_valid=np.ndarray.swapaxes(x_val,1,3);

# x_train=np.ndarray.reshape(x_tr1,x_tr1.shape[0],2,x_tr1.shape[1],x_tr1.shape[2]);
# x_valid=np.ndarray.reshape(x_val,x_val.shape[0],2,x_val.shape[1],x_val.shape[2]);

y_train=np.ndarray.reshape(y_tr1,y_tr1.shape[0],1);
y_valid=np.ndarray.reshape(y_val,y_val.shape[0],1);

# m=y.shape[0]; # size of total data
#
# tr_end=int(0.66*m); # keep a 66,33 ratio for train test
#
# x_train=x[0:tr_end-1,:,:,:];
# y_train=y[0:tr_end-1];
#
# x_valid=x[tr_end:m-1,:,:,:];
# y_valid=y[tr_end:m-1];

e2e_arch = [
    ['e2e',  # e2e layer
     {'n_filters': 16,  # 32 feature maps
      'kernel_h': x_train.shape[2], 'kernel_w': x_train.shape[3]  # Sliding cross filter of size h x 1 by 1 x w
      }
     ],
    ['e2n', {'n_filters': 16, 'kernel_h': x_train.shape[2], 'kernel_w': x_train.shape[3]}],
    ['dropout', {'dropout_ratio': 0.5}],
    ['relu', {'negative_slope': 0.33}],
    ['fc', {'n_filters': 16}],
    ['fc', {'n_filters': 16}],
    ['relu', {'negative_slope': 0.33}],
    ['out', {'n_filters': 1}]
]

siam_A = BrainNetCNN('e2e', e2e_arch)  # Create BrainNetCNN model
siam_B = BrainNetCNN('e2e', e2e_arch)

e2e_siamese = [siam_A, siam_B]

'''# Debug code

xt = np.zeros((190, 2, 110, 110), dtype='float32')  # dummy data
xt[100:190, 0:1, :, :] = np.ones((90, 1, 110, 110), dtype='float32')
#
xv = np.zeros((90, 2, 110, 110), dtype='float32')  # dummy data
xv[50:90, 0:1, :, :] = np.ones((40, 1, 110, 110), dtype='float32')
# 
#
yt = np.zeros((190, 1), dtype='float32')  # Only first 100 xt and xt2 match
yt[0:100] = np.ones((100, 1), dtype='float32')
#
yv = np.zeros((90, 1), dtype='float32')  # Only first 50 xv and xv2 match
yv[0:50] = np.ones((50, 1), dtype='float32')
#
siam_A.fit_siamese(xt, yt, xv, yv)  # Train (regress only on class 0)
'''

x_v = np.ones((10, 2, 110, 110), dtype='float32')
y_v = np.ones((10,1), dtype='float32')

x_v[0:10,:,:,:]=x_valid[0:10,:,:,:]
y_v[0:10,:]=y_valid[0:10,:]

x_v1 = np.ones((1, 1, 110, 110),dtype='float32')
y_v1 = np.ones((1, 1),dtype='float32')

#siam_A.fit(x_v1, y_v1, x_v1, y_v1)
#siam_A.fit_siamese(x_v, y_v, x_v, y_v)

siam_A.fit_siamese(x_train, y_train, x_valid, y_valid)

siam_A.fit_siamese(x_valid, y_valid, x_valid, y_valid)

siam_A.fit_siamese(x_valid[0:50,:,:,:], y_valid[0:50,:], x_valid[0:50,:,:,:], y_valid[0:50,:])



siam_A.fit_siamese(x_v, y_v, x_v, y_v)  # Train (regress only on class 0)
# siam_A.fit_siamese(x_train, y_train, x_valid, y_valid)  # Train (regress only on class 0)



siam_A.fit(x_train, y_train[:, 0], x_valid, y_valid[:, 0])  # Train (regress only on class 0)
preds = siam_A.predict(x_test)  # Predict labels of test data
print("Correlation:", pearsonr(preds, y_test[:, 0])[0])

siam_A.plot_iter_metrics()

print("Correlation:", pearsonr(preds, y_test[:, 0])[0])
