################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import json
from pprint import pprint

# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested

    # Initializing the result json dict
    results = {
        '7c': {},
        '8f': {},
    }
    lr_list = 10 ** np.linspace(-6, 2, 9)
    for lr in lr_list:
        kwargs = {'hidden_dims': [128], 'lr': lr, 'use_batch_norm': False, 'batch_size': 128, 'epochs': 10,  'seed': 42, 'data_dir': 'data/'}
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**kwargs)
        results['7c'][str(lr)] = {}
        results['7c'][str(lr)]['train_accuracies'] = logging_info['training_acc']
        results['7c'][str(lr)]['val_accuracies'] = val_accuracies
        results['7c'][str(lr)]['test_accuracy'] = test_accuracy
        results['7c'][str(lr)]['training_loss'] = logging_info['train_loss']
        results['7c'][str(lr)]['validation_loss'] = logging_info['validation_loss']
        
    
    
    hidden_dims = [[128], [256, 128], [512, 256, 128]]
    for hidden_dim in hidden_dims:
        kwargs = {'hidden_dims': hidden_dim, 'lr': 0.1, 'use_batch_norm': False, 'batch_size': 128, 'epochs': 20,  'seed': 42, 'data_dir': 'data/'}
        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**kwargs)
        results['8f'][str(hidden_dim)] = {}
        results['8f'][str(hidden_dim)]['train_accuracies'] = logging_info['training_acc']
        results['8f'][str(hidden_dim)]['val_accuracies'] = val_accuracies
        results['8f'][str(hidden_dim)]['test_accuracy'] = test_accuracy
        results['8f'][str(hidden_dim)]['training_loss'] = logging_info['train_loss']
        results['8f'][str(hidden_dim)]['validation_loss'] = logging_info['validation_loss']
    

    json_object = json.dumps(results, indent=6)
    with open(results_filename, "a") as f:
        f.write(json_object)

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, "r") as f:
        results = json.load(f)
    
    # lr_list = 10 ** np.linspace(-6, 2, 9)
    hidden_dims = [[128], [256, 128], [512, 256, 128]]
    Y = []
    X = []
    for hidden_dim in hidden_dims:
        # pprint(results['7c'][str(lr)]['training_loss'])
        # pprint(np.array(results['7c'][str(lr)]['training_loss']).shape)
        Y.append(np.array(results['8f'][str(hidden_dim)]['val_accuracies']))
        X.append(np.array(results['8f'][str(hidden_dim)]['train_accuracies']))


    plt.figure()
    for i in range(3):
        plt.title(f'Validation Accuracy PyTorch MLP')
        plt.plot(Y[i], label=str(hidden_dims[i]))
        # plt.plot(X[i], label="Train_acc")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"plot_acc_all.png")
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.json' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)