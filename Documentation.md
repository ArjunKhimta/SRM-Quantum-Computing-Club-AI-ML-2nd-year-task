# SRM-Quantum-Computing-Club-AI-ML-2nd-year-task
This Python performs the following Functions on the provided data set
Data Preprocessing
1. Cleaning the Dataset:

Loads the dataset with pandas and converts columns (Change, Percentage Change, 365 Day Percentage Change, 30 Day Percentage Change) to numeric types.
Fills missing values in these columns with their respective means to maintain data integrity.

2. Encoding Categorical Variables:

One-hot encodes categorical variables (Industry, Series), dropping the first category to prevent multicollinearity.
Regression
1. Building the Model:

Uses Last Traded Price as the target for regression.
Excludes non-numeric columns (Company Name, Symbol) from features. A Random Forest Regressor is trained due to its robustness.

2. Evaluating the Model:

Measures performance using:
Root Mean Squared Error (RMSE): Reflects the average magnitude of errors.
Mean Absolute Error (MAE): Shows the average absolute difference between predictions and actual values.
Classification

1. Creating a Binary Target:

Converts Percentage Change into a binary target (Target Change), with 1 for positive changes and 0 for others.

2. Applying Models:

SVM (Support Vector Machine): Trains a linear SVM classifier to predict the binary target.
Random Forest Classifier: Trains another classifier for comparison.

3. Evaluating Models:

Both models are assessed using:
Accuracy: Proportion of correct predictions.
Precision: True positives among positive predictions.
Recall: True positives among actual positives.
F1 Score: Harmonic mean of precision and recall.
Comparing Results

 Random Forest vs. SVM and Regression Models:

Regression: Compares Random Forest Regressor performance with RMSE and MAE.
Classification: Compares Random Forest and SVM results on accuracy, precision, recall, and F1 score.
The code preprocesses data, builds regression and classification models, and evaluates their performance.
