# Author: Usman Sadiq
# Batch file executing all GCN vs ANN(fMRI) work

# Performing a series of tasks on ABIDE(fMRI) data
# Task 1:
#       Compare GCN with ANN on ABIDE data

import os, sys
import numpy as np
import save_ABIDEnws
import gcn_fast_main
import cPickle as pickle
import matplotlib as mpl
from scipy.stats.stats import pearsonr

import plot_tools

sys.path.append("/home/usmansadiq/caffe/python")
import caffe
import matplotlib.pyplot as plt

## SIAMESE ANN
## Starting with implementing siamese structure for ANN

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))  # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN


fname='/proj/dayanelab/users/musman/DL/DeepLearn/fMRI/gcn_metric_learning/conn_train_small';
# save_ABIDEnws.ABIDE_save(60, fname);

#
subs=450
#save_ABIDEnws.ABIDE_save(subs, fname)
# This number must be divisible by 15,
# Also, the 'lines' variable on line 49 in abide_utils must be lines = subs + 1

samples=200;
batch_size=10;

num_batch=samples/batch_size;

epochs=5;

rs = 222


'''plt1=[0,1,2,3,4]
plt2=[1,2,3,4,5]

kwargs = {"plots": [plt1, plt2], "titles": ['ANN','CNN']}
plot_tools.IPlot(farg=2,**kwargs)'''

print("Random state is %d" % rs)
prng = np.random.RandomState(rs)

np.random.seed(seed=222)

# Loading training data to compare:
data = np.load(fname+'.npz');

injury = ConnectomeInjury()  # Generate train/test synthetic data.
x_sim, y_sim = injury.generate_injury()

x_tr1=np.array(data['name1'],dtype=np.float32);
y_tr1=np.array(data['name2'],dtype=np.float32);

x_val=np.array(data['name3'],dtype=np.float32);
y_val=np.array(data['name4'],dtype=np.float32);

# Reshape data for ann

x_train=np.ndarray.swapaxes(x_tr1,1,3);
x_valid=np.ndarray.swapaxes(x_val,1,3);

y_train=np.ndarray.reshape(y_tr1,y_tr1.shape[0],1);
y_valid=np.ndarray.reshape(y_val,y_val.shape[0],1);

mgcn_train, mgcn_test=gcn_fast_main.gcn_custom_run(fname,epochs,batch_size);
gcn_train, gcn_test=gcn_fast_main.gcn_run(fname,epochs,batch_size);


kwargs = {"plots": [gcn_train,mgcn_train], "titles": ['GCN train','Customized GCN train']}
plot_tools.IPlot(farg=2,**kwargs);

kwargs = {"plots": [gcn_test,mgcn_test], "titles": ['GCN test','Customized GCN test']}
plot_tools.IPlot(farg=2,**kwargs);

kwargs = {"plots": [gcn_train,gcn_test], "titles": ['GCN train','GCN test']}
plot_tools.IPlot(farg=2,**kwargs);

kwargs = {"plots": [mgcn_train,mgcn_test], "titles": ['MGCN train','MGCN test']}
plot_tools.IPlot(farg=2,**kwargs);

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

lean_arch = [
    ['e2e',  # e2e layer
     {'n_filters': 4,  # 4 feature maps
      'kernel_h': x_train.shape[2], 'kernel_w': x_train.shape[3]  # Sliding cross filter of size h x 1 by 1 x w
      }
     ],
    ['e2e',  # e2e layer
     {'n_filters': 16,  # 4 feature maps
      'kernel_h': x_train.shape[2], 'kernel_w': x_train.shape[3]  # Sliding cross filter of size h x 1 by 1 x w
      }
     ],
    ['e2n', {'n_filters': 16, 'kernel_h': x_train.shape[2], 'kernel_w': x_train.shape[3]}],
    ['dropout', {'dropout_ratio': 0.5}],
    ['relu', {'negative_slope': 0.33}],
    ['fc', {'n_filters': 2}],
    ['relu', {'negative_slope': 0.33}],
    ['out', {'n_filters': 1}]
]

exp_arch = [
    ['fc', {'n_filters': 2}],
    ['relu', {'negative_slope': 0.33}],
    ['out', {'n_filters': 1}]
]

siam_A = BrainNetCNN('ann_siam', exp_arch)  # Create BrainNetCNN model

siam_A.pars['max_iter']=epochs*num_batch # running for correct number of iterations
siam_A.pars['snapshot'] = epochs*num_batch
siam_A.pars['train_batch_size'] = batch_size
siam_A.pars['test_batch_size'] = batch_size

#siam_A.fit_siamese(x_train, y_train, x_valid, y_valid)
ann_train=siam_A.fit_siamese(x_train, y_train, x_valid, y_valid)

print("Finished executing ANN")

## SIAMESE GCN
## Starting with implementing siamese structure for GCN

gcn_train, gcn_test=gcn_fast_main.gcn_run(fname,epochs,batch_size);

print("Finished executing GCN");

#plot_tools.IPlot(1,2);

kwargs = {"plots": [ann_train.train_metrics,gcn_train], "titles": ['ANN train','GCN train']}
plot_tools.IPlot(farg=2,**kwargs);

kwargs = {"plots": [ann_train.test_error,gcn_test], "titles": ['ANN test','GCN test']}
plot_tools.IPlot(farg=2,**kwargs);







