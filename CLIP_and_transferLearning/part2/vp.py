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

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import random

class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.pad_size = args.prompt_size
        self.image_size = args.image_size
        # pad_size = 3
        # image_size = 10

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        self.pad_down = nn.Parameter(torch.rand(1, 3, self.pad_size, self.image_size))
        self.pad_up   = nn.Parameter(torch.rand(1, 3, self.pad_size, self.image_size))
        self.pad_left = nn.Parameter(torch.rand((1, 3, self.image_size - 2 * self.pad_size, self.pad_size)))
        self.pad_right= nn.Parameter(torch.rand((1, 3, self.image_size - 2 * self.pad_size, self.pad_size)))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        # [:, :, :pad, :img_size]
        x[:, :, :self.pad_size, :self.image_size] += self.pad_up
        # [:, :, img_size - pad:img_size, img_size]
        x[:, :, self.image_size-self.pad_size:self.image_size, :self.image_size] += self.pad_down
        # # [:, :, pad:img_size-pad, :pad]
        x[:, :, self.pad_size:self.image_size-self.pad_size,:self.pad_size] += self.pad_left
        # # [:, :, pad:img_size-pad, img_size-pad:img_size]
        x[:, :, self.pad_size:self.image_size-self.pad_size, self.image_size-self.pad_size:self.image_size] += self.pad_right

        return x
        #######################
        # END OF YOUR CODE    #
        #######################

    def test_padding(self):
        """
        Visualizes the padding that you defined.
        """
        # Create a dummy image
        x = torch.zeros((1, 3, self.image_size, self.image_size))
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.pause(3)
        # Add the prompt to the image
        x = self.forward(x)
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.pause(3)


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.
        self.prompt_size = args.prompt_size
        self.image_size = args.image_size
        # self.prompt_size = 23
        # self.image_size = 120

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.patch = nn.Parameter(torch.randn((1, 3, self.prompt_size, self.prompt_size)))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        x[:, :, :self.prompt_size, :self.prompt_size] += self.patch


        return x
        #######################
        # END OF YOUR CODE    #
        #######################
    def test_patch(self):
        """
        Visualizes the padding that you defined.
        """
        # Create a dummy image
        x = torch.zeros((1, 3, self.image_size, self.image_size))
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.pause(3)
        # Add the prompt to the image
        x = self.forward(x)
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.pause(3)


class RandomPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a random patch in the image.
    For refernece, this prompt should look like Fig 2(b) in the PDF.
    """
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can be defined as self.patch) of size [prompt_size, prompt_size]
        # that is located at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.prompt_size = args.prompt_size
        self.image_size = args.image_size
        # self.prompt_size = 4
        # self.image_size = 120

        self.random_patch = nn.Parameter(torch.randn((1, 3, self.prompt_size, self.prompt_size)))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - Note that, here, you need to place the patch at a random location
        #   and not at the top-left corner.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        i = random.randint(0, self.image_size - self.prompt_size)
        j = random.randint(0, self.image_size - self.prompt_size)

        x[:, :, i:i+self.prompt_size, j:j+self.prompt_size] += self.random_patch

        return x        
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def test_patch(self):  
        """
        Visualizes the padding that you defined.
        """
        # Create a dummy image
        x = torch.zeros((1, 3, self.image_size, self.image_size))
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.pause(2)
        for i in range(5):
            y = self.forward(x)
            # Visualize the image
            plt.imshow(y[0].permute(1, 2, 0).detach().numpy())
            plt.pause(0.5)
            plt.clf()

class CustomPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(CustomPrompter, self).__init__()

        # assert isinstance(args.image_size, int), "image_size must be an integer"
        # assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.
        self.prompt_size = args.square_size
        self.image_size = args.image_size
        # self.prompt_size = 30
        # self.image_size = 224

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn

        self.patch = nn.Parameter(torch.randn((1, 3, self.prompt_size, self.prompt_size)))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        # stride =  self.image_size - self.prompt_size
        
        n = random.randint(1, 30)
        for _ in range(n):
            i = random.randint(0, self.image_size - self.prompt_size)
            j = random.randint(0, self.image_size - self.prompt_size)
            x[:, :, i:i+self.prompt_size, j:j+self.prompt_size] += self.patch

        return x  

        #######################
        # END OF YOUR CODE    #
        #######################
    def test_patch(self):
        """
        Visualizes the padding that you defined.
        """
        # Create a dummy image
        x = torch.zeros((1, 3, self.image_size, self.image_size))
        x = self.forward(x)
        # Visualize the image
        plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
        plt.savefig('custom_prompt.png')
        plt.pause(3)

if __name__ == "__main__":
    args = {}
    # Pad = PadPrompter(args)
    # Pad.test_padding()
    # PatchPrompter = FixedPatchPrompter(args)
    # PatchPrompter.test_patch()
    # RandomPatch = RandomPatchPrompter(args)
    # RandomPatch.test_patch()
    custom = CustomPrompter(args)
    custom.test_patch()
