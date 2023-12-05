import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(input_file):
    df = pd.read_csv(input_file, header=None)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y

# Split data
def split_data(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    return classifier

# Evaluate model
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = 100.0 * (y_test == y_pred).sum() / X_test.shape[0]
    print("Accuracy of the classifier =", round(accuracy, 2), "%")
    return y_pred

# Display confusion matrix
def display_confusion_matrix(y_test, y_pred):
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)
    sns.heatmap(confusion_mat, annot=True, fmt='d')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()

# Display classification report
def display_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

# Main function
def main():
    input_file = 'wine.txt'
    X, y = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(X, y)
    classifier = train_model(X_train, y_train)
    y_pred = evaluate_model(classifier, X_test, y_test)
    display_confusion_matrix(y_test, y_pred)
    display_classification_report(y_test, y_pred)

if __name__ == "__main__":
    main()