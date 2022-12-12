################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import numpy as np
import os
from copy import deepcopy
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size, n_classes = predictions.shape
    predictions = torch.argmax(predictions, dim=1)
    conf_mat = torch.zeros((n_classes, n_classes))

    # Building confusion matrix
    for i in range(batch_size):
        pred_class = predictions[i]
        target_class = targets[i]
        conf_mat[pred_class, target_class] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.0):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    metrics = {}
    metrics['accuracy']  = torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)
    metrics['precision'] = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis=1))
    metrics['recall']    = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, axis=0))
    metrics['f1_score']  = (1 + beta**2) * (metrics['precision'] * metrics['recall']) / ((beta**2) * metrics['precision'] + metrics['recall'])

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, beta=1, num_classes=10 ):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_module = nn.CrossEntropyLoss().to(device)
    n_inputs= 32 *32 * 3
    conf_matrix = torch.zeros((num_classes, num_classes))
    losses = []
    for i, data in enumerate(data_loader):
        image_batch, label_batch = data
        batch_size = image_batch.shape[0]
        image_batch = torch.reshape(image_batch, shape=(batch_size, n_inputs))
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        with torch.no_grad():
            out = model.forward(image_batch)
            loss = loss_module(out, label_batch)
            
        conf_matrix += confusion_matrix(out, label_batch)
        losses.append(loss.item())
        metrics = confusion_matrix_to_metrics(conf_matrix,beta)
        metrics["conf_matrix"] = conf_matrix
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics, np.mean(losses)


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir, beta=1, n_classes = 10,n_inputs = 32 * 32 * 3, verbose=True):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    print(f"""
            Device: {device}
            Learning rate: {lr},
            Beta: {beta},
            Hidden dimensions: {hidden_dims},
            Batch size: {batch_size}, 
            Epochs: {epochs}
            """)

    train_loader = cifar10_loader["train"]
    validation_loader = cifar10_loader["validation"]
    test_loader = cifar10_loader["test"]

    # TODO: Initialize model and loss module
    model = MLP(n_inputs, hidden_dims, n_classes, use_batch_norm)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    loss_module.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    print("Model Initialized")
    if verbose:
        print(model)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    best_acc = 0.0
    best_model = None

    for epoch in range(epochs):

        model.train()
        conf_matrix = torch.zeros((n_classes, n_classes))
        epoch_losses = []
        for i, data in enumerate(train_loader):
            image_batch, label_batch = data
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            image_batch = torch.reshape(image_batch, shape=(batch_size, n_inputs))

            #Forward pass
            out = model(image_batch)
            loss = loss_module(out, label_batch)
            conf_matrix += confusion_matrix(out, label_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        training_loss.append(epoch_loss)
        train_metrics = confusion_matrix_to_metrics(conf_matrix)
        training_acc.append(train_metrics["accuracy"].item())

        # Validation
        model.eval()
        val_metrics, val_loss = evaluate_model(model, validation_loader, beta, n_classes)
        validation_acc.append(val_metrics["accuracy"].item())
        validation_loss.append(val_loss)
        if val_metrics["accuracy"].item() > best_acc:
            best_acc = val_metrics["accuracy"].item()
            best_model = deepcopy(model)
        
        if verbose:
            print(f"------ Epoch {epoch + 1}: ------ ")
            print(f"  Training loss: {epoch_loss:.3f}, Accuracy: {training_acc[-1]:.3f}")
            print(f"  Validation loss: {val_loss:.3f} ---  Accuracy: {validation_acc[-1]:.3f}")
            print(f"     ")

    # # TODO: Test best model
    test_metrics, _ = evaluate_model(best_model, test_loader, beta)
    test_accuracy = test_metrics["accuracy"].item()
    print(f"TEST Accuracy: {test_accuracy:.3f} for beta == 1 ")

    # # TODO: Add any information you might want to save for plotting
    logging_info = {
        "train_loss": training_loss,
        "training_acc": training_acc,
        "validation_loss": validation_loss,
        "validation_acc": validation_acc,
        "confusion_matrix" : test_metrics['conf_matrix'],
        "test_metrics": test_metrics
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, validation_acc, test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    for beta in [0.1, 1, 10]:
        model, validation_acc, test_accuracy, logging_info = train(beta=beta,**kwargs)
        # Feel free to add any additional functions, such as plotting of the loss curve here
        print(f"Beta: {beta} -- Test Accuracy {test_accuracy}\n")

    with open("logging_info_torch_bbb.pickle", "wb") as handle:
        pickle.dump(logging_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Training finished and saved")
