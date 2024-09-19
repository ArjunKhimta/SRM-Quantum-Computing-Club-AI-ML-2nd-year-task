import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/Users/arjunkhimta/Downloads/nifty_500.csv")

data['Change'] = pd.to_numeric(data['Change'], errors='coerce')
data['Percentage Change'] = pd.to_numeric(data['Percentage Change'], errors='coerce')
data['365 Day Percentage Change'] = pd.to_numeric(data['365 Day Percentage Change'], errors='coerce')
data['30 Day Percentage Change'] = pd.to_numeric(data['30 Day Percentage Change'], errors='coerce')

data['Change'] = data['Change'].fillna(data['Change'].mean())
data['Percentage Change'] = data['Percentage Change'].fillna(data['Percentage Change'].mean())
data['365 Day Percentage Change'] = data['365 Day Percentage Change'].fillna(data['365 Day Percentage Change'].mean())
data['30 Day Percentage Change'] = data['30 Day Percentage Change'].fillna(data['30 Day Percentage Change'].mean())

data = pd.get_dummies(data, columns=['Industry', 'Series'], drop_first=True)

X_reg = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y_reg = data['Last Traded Price']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = rf_regressor.predict(X_test_reg)

rmse_reg = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"Regression Results:\nRMSE: {rmse_reg}\nMAE: {mae_reg}")

data['Target Change'] = (data['Percentage Change'] > 0).astype(int)

X_clf = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol', 'Percentage Change', 'Target Change'])
y_clf = data['Target Change']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train_clf, y_train_clf)
y_pred_svm = svm_clf.predict(X_test_clf)

accuracy_svm = accuracy_score(y_test_clf, y_pred_svm)
precision_svm = precision_score(y_test_clf, y_pred_svm)
recall_svm = recall_score(y_test_clf, y_pred_svm)
f1_svm = f1_score(y_test_clf, y_pred_svm)

print(f"SVM Results:\nAccuracy: {accuracy_svm}\nPrecision: {precision_svm}\nRecall: {recall_svm}\nF1 Score: {f1_svm}")

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf = rf_clf.predict(X_test_clf)

accuracy_rf = accuracy_score(y_test_clf, y_pred_rf)
precision_rf = precision_score(y_test_clf, y_pred_rf)
recall_rf = recall_score(y_test_clf, y_pred_rf)
f1_rf = f1_score(y_test_clf, y_pred_rf)

print(f"Random Forest Classifier Results:\nAccuracy: {accuracy_rf}\nPrecision: {precision_rf}\nRecall: {recall_rf}\nF1 Score: {f1_rf}")
