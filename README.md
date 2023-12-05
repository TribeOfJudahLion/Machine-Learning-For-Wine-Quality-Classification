<br/>
<p align="center">
  <h3 align="center">Uncork the Power of AI: Wine Quality Classification Re-imagined</h3>

  <p align="center">
    Savor the taste of data-driven excellence – Vintage meets AI for impeccable wine quality assessment!
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Scenario:

A data science team at a winery wants to improve the quality control process for their wine production. They have historical data on wines that were previously tested and rated by experts. The dataset contains various physicochemical attributes of the wines and a quality rating. The goal is to predict the quality of wine (as a classification problem) based on its attributes, thus automating the quality assessment process and ensuring consistent product quality.

The team decides to use a machine learning approach to classify the wines into different quality categories. They choose a Decision Tree Classifier for its interpretability and because it works well with categorical data.

### Solution:

The Python script provided by the team follows these steps to address the problem:

#### 1. Loading the Data:
- `load_data(input_file)`: The dataset (`wine.txt`) is loaded into a pandas DataFrame. Features (`X`) and labels (`y`) are separated. The features are the physicochemical attributes, and the labels are the wine quality ratings.

#### 2. Splitting the Dataset:
- `split_data(X, y)`: The dataset is split into a training set (75%) and a test set (25%). This allows the model to learn from the training data and then be evaluated on unseen data to ensure it can generalize well.

#### 3. Training the Model:
- `train_model(X_train, y_train)`: A Decision Tree Classifier is trained with the training set. The model learns to associate the features with the quality ratings.

#### 4. Evaluating the Model:
- `evaluate_model(classifier, X_test, y_test)`: The trained model is used to predict the quality of the wines in the test set. The predictions are compared to the actual ratings to calculate the accuracy of the model.

#### 5. Displaying Results:
- `display_confusion_matrix(y_test, y_pred)`: A confusion matrix is generated and visualized to show the model's performance across different classes. This helps the team see where the model is confused and potentially improve it.
- `display_classification_report(y_test, y_pred)`: A classification report is printed, providing precision, recall, f1-score, and support for each class. This comprehensive performance analysis helps the team understand the strengths and weaknesses of the model.

#### 6. Main Execution:
The `main()` function orchestrates the flow of data through the model pipeline: loading data, splitting, training, evaluating, and displaying results. This function is executed when the script is run.

### Outcome:
The execution of this script enables the winery to:
- Automate the prediction of wine quality with a high level of accuracy.
- Identify which attributes of wine are most predictive of quality.
- Reduce the reliance on subjective human taste tests.
- Enhance the quality control process with a consistent and objective assessment tool.

By analyzing the output, the winery can continue to refine their machine learning model, potentially exploring other algorithms, feature engineering, or hyperparameter tuning to improve prediction accuracy.

This script is a Python program designed to perform data loading, model training, evaluation, and visualization in a machine learning context, specifically using decision trees. Let's go through it step by step.

### 1. Importing Libraries
- `numpy` and `pandas` are used for data manipulation.
- `sklearn` (scikit-learn) provides tools for data splitting, model training, and evaluation.
- `matplotlib.pyplot` and `seaborn` are used for data visualization.

### 2. Function `load_data(input_file)`
- Loads data from a CSV file (`input_file`).
- Assumes no header in the CSV file.
- Splits data into features (`X`) and labels (`y`).

### 3. Function `split_data(X, y)`
- Splits the dataset into training and testing sets.
- `test_size=0.25` means 25% of the data is used for testing.
- `random_state=5` ensures reproducibility by setting a seed for the random number generator.

### 4. Function `train_model(X_train, y_train)`
- Initializes a `DecisionTreeClassifier`.
- Trains it with the training data (`X_train`, `y_train`).
- Returns the trained classifier.

### 5. Function `evaluate_model(classifier, X_test, y_test)`
- Uses the trained classifier to make predictions on the test set (`X_test`).
- Calculates and prints the accuracy of the model.
- Returns the predictions (`y_pred`).

### 6. Function `display_confusion_matrix(y_test, y_pred)`
- Generates a confusion matrix from the true labels (`y_test`) and predictions (`y_pred`).
- Visualizes the confusion matrix using `seaborn`'s heatmap.

### 7. Function `display_classification_report(y_test, y_pred)`
- Prints a classification report that includes metrics like precision, recall, and F1-score.

### 8. The `main` Function
- Defines the flow of the program:
    - Load the data from 'wine.txt'.
    - Split the data into training and test sets.
    - Train the decision tree classifier.
    - Evaluate the model.
    - Display confusion matrix and classification report.

### 9. Execution Trigger
- `if __name__ == "__main__":` ensures that `main()` is called only if the script is executed as a standalone program (not imported as a module).

### Key Points
- The script focuses on classification using a decision tree.
- It covers the full workflow from data loading to model evaluation.
- The use of visualization (confusion matrix) and detailed metrics (classification report) aids in understanding model performance.
- The script is set up for reproducibility and can be easily adapted for other classification tasks by changing the dataset and possibly the classifier.

The output includes a confusion matrix visualization, accuracy of the classifier, numerical confusion matrix data, and a classification report. Let's break down each part:

### 1. Confusion Matrix Visualization
The heatmap displays the confusion matrix, which is a table used to describe the performance of a classification model on a set of test data for which the true values are known. In this matrix:
- The x-axis represents the predicted classes.
- The y-axis represents the actual classes.
- The color intensity and the number represent the count of instances.

In the given matrix:
- There are three classes (0, 1, 2).
- Diagonal cells (17, 12, 12) represent correct predictions.
- Non-diagonal cells show the misclassifications (e.g., 2 instances of class 0 were predicted as class 1).

### 2. Accuracy of the Classifier
- It is calculated as the number of correct predictions divided by the total number of predictions.
- The classifier's accuracy is 91.11%, which means that it correctly predicted the class 91.11% of the time.

### 3. Numerical Confusion Matrix
- It's a 3x3 matrix because there are three classes.
- Each cell shows the number of predictions made by the classifier as follows:
  - `[[17  2  0] [ 1 12  1] [ 0  0 12]]`
  - True class 0: 17 correct predictions, 2 incorrect as class 1, 0 as class 2.
  - True class 1: 1 incorrect as class 0, 12 correct, 1 incorrect as class 2.
  - True class 2: all 12 correct, none incorrect.

### 4. Classification Report
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to a low false positive rate.
  - Class 1: 0.94
  - Class 2: 0.86
  - Class 3: 0.92
- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all observations in actual class. High recall relates to a low false negative rate.
  - Class 1: 0.89
  - Class 2: 0.86
  - Class 3: 1.00
- **F1-score**: The weighted average of Precision and Recall. Useful when the class distribution is uneven. Scores range from 0 to 1, with 1 being perfect precision and recall.
  - Class 1: 0.92
  - Class 2: 0.86
  - Class 3: 0.96
- **Support**: The number of actual occurrences of the class in the specified dataset. For class balance insights.
  - Class 1: 19
  - Class 2: 14
  - Class 3: 12
- **Accuracy**: Overall, how often the classifier is correct (given above as 91.11%).
- **Macro avg**: Average precision, recall, and F1-score between classes, treating all classes equally.
- **Weighted avg**: Weighted average precision, recall, and F1-score between classes, considering support.

The high values of precision, recall, and F1-score across the classes indicate that the classifier performs well for this dataset.

## Built With

This project is built with a variety of powerful tools and libraries that facilitate machine learning, data manipulation, and visualization in Python:

- **[NumPy](https://numpy.org/)**: A fundamental package for scientific computing with Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. NumPy arrays are used for handling the data and performing numerical operations.

- **[Pandas](https://pandas.pydata.org/)**: An open-source data analysis and manipulation tool, built on top of the Python programming language. It offers data structures and operations for manipulating numerical tables and time series, which is essential for handling datasets in machine learning.

- **[Scikit-learn](https://scikit-learn.org/stable/)**: A free software machine learning library for the Python programming language. It features various classification, regression, and clustering algorithms including:
    - `model_selection.train_test_split`: For splitting the dataset into training and test sets.
    - `DecisionTreeClassifier`: A decision tree algorithm for classification tasks.
    - `confusion_matrix` and `classification_report`: For model evaluation metrics.

- **[Matplotlib](https://matplotlib.org/)**: A plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits. Used here to visualize the confusion matrix.

- **[Seaborn](https://seaborn.pydata.org/)**: A Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. In this project, Seaborn is utilized to enhance the confusion matrix visualization with a heatmap.

The above libraries are essential for the machine learning workflow that includes data loading, preprocessing, model training, prediction, and evaluation. Each of these steps is carried out by specific functions from the libraries, providing a robust framework for developing machine learning models.

## Getting Started

This section is designed to guide you through the setup process so you can get this project up and running on your local machine for development and testing purposes.

#### Prerequisites

Before you begin, ensure you have the following installed:
- Python (3.6 or higher)
- pip (Python package installer)

This project relies on several Python libraries listed below. You can install them using `pip`.

#### Installation

1. **Clone the repository**

   First, clone this repository to your local machine using Git.
   ```
   git clone https://github.com/your-username/your-repo-name.git
   ```
   Navigate to the cloned directory.
   ```
   cd your-repo-name
   ```

2. **Set up a virtual environment (optional but recommended)**

   It's a good practice to create a virtual environment for your project. This keeps your project dependencies separated from your system-wide Python packages.
   ```
   python -m venv venv
   ```
   Activate the virtual environment.
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

3. **Install the required packages**

   Install all the required packages using `pip` and the provided `requirements.txt` file.
   ```
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
   *If you have a `requirements.txt` file, use the following command instead:*
   ```
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**

   Ensure you have the dataset file named `wine.txt` in the project directory. This file should be in CSV format and contain the data you wish to analyze.

#### Running the Project

Once installation is complete, you're ready to run the project.

1. **Running the script**

   Execute the main script with the following command:
   ```
   python your-script-name.py
   ```

   Replace `your-script-name.py` with the actual name of your Python script.

2. **Interpreting the output**

   The script will output the accuracy of the classifier, display the confusion matrix, and the classification report in your terminal. To fully understand the results, refer to the documentation of the confusion matrix and classification report provided by scikit-learn.

#### Troubleshooting

- If you encounter any issues with package versions, try updating the packages to their latest versions using `pip`:
  ```
  pip install --upgrade package-name
  ```
- Ensure that the `wine.txt` dataset is in the correct format as expected by the script.

#### Support

If you run into any issues or have questions, please file an issue on the GitHub repository, and a maintainer will help you troubleshoot.

#### Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Check the repository for a `CONTRIBUTING.md` file for guidelines on how to contribute.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/TribeOfJudahLion/Machine-Learning-For-Wine-Quality-Classification /issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/TribeOfJudahLion/Machine-Learning-For-Wine-Quality-Classification /blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TribeOfJudahLion/Machine-Learning-For-Wine-Quality-Classification /blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion) - **

## Acknowledgements

* []()
* []()
* []()
