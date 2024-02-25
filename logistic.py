from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            if i % 199 == 0:
                print('Iteration: %d, Cost: %f' % (i, cost))
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

# Calculate the confusion matrix
def calculate_confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    return confusion_matrix

# plot the confusion matrix
def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Average Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# k-fold cross validation
# This method splits the data into k equal parts and uses k-1 parts for training and 1 part for testing
def k_fold_cross_validation(X, y, model, k=10):
    num_samples = X.shape[0]
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    total_cm = np.zeros((len(np.unique(y)), len(np.unique(y))))

    for i in range(k):
        # Generate the indices for the test set
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        # Generate the indices for the train set
        train_indices = np.concatenate((indices[:i*fold_size], indices[(i+1)*fold_size:]))
        # Create the train and test sets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Train the model
        model.train(X_train, y_train)
        # Predict the labels for the test set
        y_pred = model.predict(X_test)
        # Calculate the accuracy, precision, and recall of the model
        accuracy = calculate_accuracy(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        recall = calculate_recall(y_test, y_pred)
        # Calculate the confusion matrix
        confusion = calculate_confusion_matrix(y_test, y_pred)
        total_cm += confusion
        # Append the scores to the respective lists
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Return the mean of the scores
    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), total_cm / k

# Use the model and print the results
model = LogisticRegreson(num_iter=200, learning_rate=0.44)
mean_accuracy, mean_precision, mean_recall, average_confusion = k_fold_cross_validation(X, y, model, k=10)
print('Mean Accuracy:', mean_accuracy)
print('Mean Precision:', mean_precision)
print('Mean Recall:', mean_recall)
print('F1 Score:', 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall))
plot_confusion_matrix(average_confusion)

