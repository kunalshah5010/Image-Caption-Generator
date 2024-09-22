# Image Caption Generation using Deep Learning

## Overview

This repository contains code for an image caption generation system using deep learning techniques. The system leverages a pretrained VGG16 model for feature extraction and a custom captioning model which was trained using LSTM for generating captions. The model is trained on the Flickr8k dataset using an attention mechanism to improve caption quality.

**Note:** While using the `VGG16` model for feature extraction provides accurate results, it's important to be mindful of memory usage. The VGG16 model can consume a significant amount of memory, potentially causing issues in resource-constrained environments. To address this, it's advised to consider using the `MobileNetV2` model for feature extraction. MobileNetV2 strikes a balance between memory efficiency and performance, making it a practical choice for scenarios with limited resources. Consequently, in my deployed app, I've opted for `MobileNetV2`.

The key components of the project include:

- Image feature extraction using a pretrained VGG16 model (Consider using MobileNetV2 for memory efficiency)
- Caption preprocessing and tokenization
- Custom captioning model architecture with attention mechanism
- Model training and evaluation
- Flask app for interactive caption generation

## About the Dataset

The [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k) is used for training and evaluating the image captioning system. It consists of 8,091 images, each with five captions describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

Download the dataset from [Kaggle](https://www.kaggle.com/adityajn105/flickr8k) and organize the files as follows:

- flickr8k
  - Images
    - (image files)
  - captions.txt


## Directory Tree

```
|   app.py
|   image-captioner.ipynb
|   mymodel.h5
|   README.md
|   tokenizer.pkl
|   index.html
```

