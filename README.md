# CIFAR10 Classification Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Parameters](https://img.shields.io/badge/Parameters-<200k-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-86.48%25-brightgreen.svg)]()

This repository implements a lightweight convolutional neural network designed for classifying images in the CIFAR-10 dataset. The model achieves 86.48% accuracy on the test set using fewer than 200k parameters.

## Model Architecture

The model is designed to be efficient and effective, employing modern deep learning techniques such as depthwise separable convolutions, dilated convolutions, attention mechanisms, and atrous spatial pyramid pooling. Below are the key components:

1. Initial Convolution Block

A standard convolution layer with Batch Normalization, ReLU activation, and Dropout.

2. Dilated Convolutional Blocks

Two blocks of depthwise separable convolutions with increasing dilation rates to expand the receptive field without increasing computation:

First block (dilation: 2, 4).

Second block (dilation: 8, 16).

3. Attention Mechanism

A squeeze-and-excitation-like module that applies global average pooling and learns attention weights for feature refinement.

4. Deconvolution and Residual Connection

A deconvolution layer is combined with a residual connection to refine features and ensure gradient flow.

5. Atrous Spatial Pyramid Pooling (ASPP)

Captures multi-scale contextual information using parallel branches with different dilation rates (6, 12, 18).

6. Final Classification Layers

Combines global average pooling, fully connected layers, and a LogSoftmax activation for classification.

## Key Features
   
[![Depthwise Separable](https://img.shields.io/badge/Feature-Depthwise%20Separable-blue.svg)]() Reduces parameter count and computation cost.

[![Dilated Convolutions](https://img.shields.io/badge/Feature-Dilated%20Convolutions-blue.svg)]() Expands the receptive field without increasing the number of parameters.

[![Attention Mechanism](https://img.shields.io/badge/Feature-Attention%20Mechanism-blue.svg)]() Focuses on important features using channel-wise weighting.

[![ASPP Module](https://img.shields.io/badge/Feature-ASPP%20Module-blue.svg)]() Enhances multi-scale feature extraction for robust classification.

[![Lightweight](https://img.shields.io/badge/Feature-Lightweight-blue.svg)]() Utilizes fewer than 200k parameters for efficiency.

## Performance

[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange.svg)]() 10 classes, 32x32 images

[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-86.48%25-brightgreen.svg)]() on the test set

[![Parameters](https://img.shields.io/badge/Parameters-<200k-green.svg)]() for efficient deployment


