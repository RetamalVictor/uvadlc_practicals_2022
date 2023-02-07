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
# Date Created: 2022-11-14
################################################################################

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader
from cifar100_utils import get_train_validation_set, get_test_set
from pprint import pprint
from copy import deepcopy
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(pretrained=True)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for _ , param in model.named_parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    model.fc.weight = nn.Parameter(torch.normal(0.0, 0.01, (num_classes, 512)))
    model.fc.bias = nn.Parameter(torch.zeros_like(model.fc.bias))
    
    enabled = set()
    for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
    print(f"Parameters to be updated:")
    pprint(f"Parameters to be updated: {enabled}")    
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None, verbose=True):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)
    train_loader =  DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=3)
    val_loader =    DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=3)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()
    model.to(device)
    loss_module.to(device)

    print(f"""
            Device: {device}
            Learning rate: {lr},
            Batch size: {batch_size}, 
            Epochs: {epochs},
            Augmentation: {augmentation_name}
            """)

    training_loss = []
    training_acc = []
    validation_acc = []
    best_acc = 0.0
    best_model = None


    for epoch in range(epochs):
        print(f"------ Epoch {epoch + 1}: ------ ")
        model.train()
        epoch_losses = []
        for i, data in enumerate(train_loader):
            image_batch, label_batch = data
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            #Forward pass
            out = model(image_batch)
            loss = loss_module(out, label_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        training_loss.append(epoch_loss)

        # Validation
        validation_acc = evaluate_model(model, val_loader, device)

        if validation_acc > best_acc:
            best_acc = validation_acc
            best_model = deepcopy(model)
            torch.save(
                model.state_dict(),
                os.path.join(
                    r"C:\Users\victo\Desktop\VU master\DL1\assignment_2\assignment2\part1\checkpoints",
                    f"{checkpoint_name}_acc_{best_acc:.2f}.pt"))

        if verbose:
            # print(f"------ Epoch {epoch + 1}: ------ ")
            print(f"  Training loss: {epoch_loss:.3f}")
            print(f"  Validation Accuracy: {validation_acc[-1]:.3f}")
            print(f"     ")

    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    model.to(device)

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    accuracy_list = []
    for i, data in enumerate(data_loader):
        image_batch, label_batch = data
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        with torch.no_grad():
            out = model.forward(image_batch)
            out = torch.argmax(out, axis=1)
        accuracy_list.append((sum(out == label_batch) / len(label_batch)).detach().cpu().item())

    accuracy = np.mean(accuracy_list)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
   # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    print("Model Initialized")
    model = get_model()

    # Get the augmentation to use
    pass

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, 'best_model', device, augmentation_name)



    # Evaluate the model on the test set
    test_set = get_test_set(data_dir)
    test_accuracy = evaluate_model(model, test_set, device)
    print(f"TEST Accuracy: {test_accuracy:.3f} for beta == 1 ")


    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
