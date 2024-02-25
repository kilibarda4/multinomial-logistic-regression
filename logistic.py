from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords

#softmax function - fits the probability of output between 0 and 1
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Logistic Regression Model that fits and predicts data
class LogisticRegreson():
    def __init__(self, num_iter, learning_rate):
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.weights = np.random.randn(1000, 4) * 0.01 # 1000 features, 4 classes
        self.bias = np.zeros(4)

    def compute_cost(self, X, y): #Cross Entropy Loss
        num_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = softmax(scores)
        correct_logprobs = -np.log(probs[range(num_samples), y])
        cost = np.sum(correct_logprobs) / num_samples
        return cost, probs

    def compute_gradients(self, X, y, probs): #Stochastic Gradient Descent
        num_samples = X.shape[0]
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples
        dweights = np.dot(X.T, dscores)
        dbias = np.sum(dscores, axis=0)
        return dweights, dbias

    def train(self, X, y):
        for i in range(self.num_iter):
            cost, probs = self.compute_cost(X, y)
            if i % 200 == 0:
                print('Iteration %d: cost = %f' % (i, cost))
            dweights, dbias = self.compute_gradients(X, y, probs)
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias
    
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)


# Load Data from csv file and turn it into lowercase letters
data = pd.read_csv('eCommerceDataset.csv').apply(lambda x: x.astype(str).str.lower())

# Replace class names with numbers
class_mapping = {
    'household': 0,
    'books': 1,
    'clothing & accessories': 2,
    'electronics': 3
}
data.iloc[:, 0] = data.iloc[:, 0].map(class_mapping).astype(int)

words = ' '.join(data.iloc[:, 1]).split() # Get the words from the description column
words = ['some_number' if word.isnumeric() else word for word in words] # Replace numbers with 'some_number' to decrease dimensionality

stop_words = set(stopwords.words('english')) # Remove stopwords from the data
stop_words.update(['-', '*', '&', '.' , '/', ':', '|']) # Additional stopwords to ignore
words = [word for word in words if word not in stop_words]

word_counts = Counter(words) # Count occurrences of each unique word
features = [word for word, count in word_counts.most_common(1000)] # Get the 1000 most common words

# Transform the descriptions into a matrix of features
def transform_to_features(descriptions, features): #Returns matrix of features
    def count_features(words):
        return [words.count(feature) for feature in features]
    
    return np.array([count_features(words) for words in descriptions])

# Split each description into words
descriptions = [description.split() for description in data.iloc[:, 1]]
X = transform_to_features(descriptions, features)
y = data.iloc[:, 0].values.astype(int)

# Get the number of total samples
num_samples = X.shape[0]

# Generate a list of indices from 0 to num_samples
indices = np.arange(num_samples)

# Shuffle the indices
np.random.shuffle(indices)

# Calculate the size of the test set
test_size = int(num_samples * 0.1)

# Split the indices into train and test indices
train_indices = indices[test_size:]
test_indices = indices[:test_size]

# Use the train and test indices to create the train and test sets
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# ADJUST THE LEARNING RATE AND NUMBER OF ITERATIONS TO GET THE BEST RESULTS
model = LogisticRegreson(num_iter=1000, learning_rate=0.44)
model.train(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy: (true positives + true negatives) / total predictions
# This method generates the ability of the model to predict the correct class
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Precision: true positives / (true positives + false positives)
# macro precision is calculated due to presence of multiple classes
# This method treats all classes equally
# To avoid dividing by zero, Laplace smoothing is used
def calculate_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    precision = 0
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_positives = np.sum((y_true != class_) & (y_pred == class_))
        precision += true_positives / (true_positives + false_positives + 1e-10)
    return precision / num_classes

# Recall: true positives / (true positives + false negatives)
# Similar approach to precision
def calculate_recall(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = 0
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_negatives = np.sum((y_true == class_) & (y_pred != class_))
        recall += true_positives / (true_positives + false_negatives + 1e-10)
    return recall / num_classes

# plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Print the results
accuracy = calculate_accuracy(y_test, y_pred)
print('Accuracy:', accuracy)

precision = calculate_precision(y_test, y_pred)
print('Precision:', precision)

recall = calculate_recall(y_test, y_pred)
print('Recall:', recall)

f1 = 2 * (precision * recall) / (precision + recall)
print('F1 Score:', f1)

plot_confusion_matrix(y_test, y_pred)