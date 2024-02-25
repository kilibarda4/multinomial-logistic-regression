# Multinomial Logistic Regression From Scratch (no, really, from scratch)

## Overview

This script contains an implementation of a Multinomial Logistic Regression model for text classification.


The model is written from scratch without use of sklearn or similar libraries.


Testing of the model is performed with 10-fold cross-validation


## Requirements

This script requires Python 3.10.2. You can check your Python version by running `python --version` in your command line.

## Dependencies

The script depends on the following Python libraries:

- numpy
- pandas
- matplotlib
- nltk

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib nltk
```

## Running the Script

To run the script, navigate to the directory containing `logistic.py` in your command line and run the following command:

```bash
python logistic.py
```

This will execute the script and output the results to your command line.

## Functionality

The script reads in a dataset, preprocesses the text data, and maps the class names to numbers. It counts the occurrences of each unique word in the descriptions and selects the 1000 most common words as features. The descriptions are then transformed into a matrix of features. The data is split into a training set and a test set (90/10 split, which can be changed according to requirements). The Logistic Regression model is trained on the training set using Stochastic Gradient Descent and the Cross Entropy Loss function. The trained model is used to predict the categories of the test set. The performance of the model is evaluated using accuracy, precision, recall, and F1 score. A confusion matrix is also plotted to visualize the performance of the model.

Please note that the script assumes that the input data is in the following format: CSV file containing class names in column one and descriptions in column two. Please refer to the comments in the script for more details on the expected input format.
