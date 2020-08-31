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

import matplotlib.pyplot as plt
import matplotlib as mpl



def IPlot(farg, **kwargs):
    num_plots = farg;
    mpl.rc("figure", facecolor="white")

    # key in kwargs


    colors = ['red', 'purple', 'black'];
    styles = ['-', '--', '-.'];
    titles = [];

    Ind=[];
    keys=kwargs.iterkeys()
    num=0;  # type: int

    '''for ii in keys:
        y = kwargs[ii];
        #titles[ii] = kwargs[2 * ii + 1];
        tmp1 = range(0, len(y)) # This is a very weird work around: python doesnt allow enumerated arguments of variable size
        if(num==0):
            tmp=tmp1;

        print(y)

        #ax = plt.plot(tmp, y[0:len(tmp)])
        if(num%2==0): # plot this array
            ax = plt.plot(tmp, y[0:len(tmp)], color=colors[num], linewidth=3, linestyle=styles[num])
            else
            titles[ii] = kwargs[2 * ii + 1];
        num+=1'''

    pltData=kwargs['plots'];
    title=kwargs['titles'];

    y = pltData[0]
    Ind=range(0,len(y))
    legs=[]

    for ii in range(num_plots):
        y=pltData[ii]
        ax = plt.plot(Ind, y[0:len(Ind)], color=colors[ii], linewidth=3, linestyle=styles[ii])
        #legs[ii]=title[ii]

    plt.xlabel('Epochs')
    plt.ylabel('Euclidean loss')
    plt.title('Siamese network - training')

    plt.legend(title)

    plt.show()

    return None
# Load data first

