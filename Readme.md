# Multinomial Logistic Regression From Scratch (no, really, from scratch)

## Overview

This script contains an implementation of a Multinomial Logistic Regression model for text classification.


The model is written from scratch without use of sklearn or similar libraries.


Testing of the model is performed with 10-fold cross-validation.


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

- Preprocessing:
    - Data is turned into lowercase
    - Stopwords are removed using a nlkt library function
    - All numbers are treated as 'someNumber' to reduce dimensionality of the problem
    - Most common 1000 words are extracted after preprocessing and used as features

- Training/Testing:
    - The model is trained using the .train function with parameters (int num_iterations, float learning_rate) which can be fine tuned to trade between speed and accuracy
    - Trained model is tested on unseen portion of dataset

- Results (printed to terminal):
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix plotted to visualize performance of the model

Please note that the script assumes that the input data is in the following format: CSV file containing class names in column one and descriptions in column two. Please refer to the provided dataset for more details on the expected input format.