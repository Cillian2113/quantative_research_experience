import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#Read csv file into df
loan_data = pd.read_csv('Task 3 and 4_Loan_Data.csv')
loan_data = loan_data.drop(columns=["customer_id"])
loan_data = loan_data.dropna()

#Split in half for training and testing
loan_training_data = loan_data.iloc[:5000]
loan_testing_data = loan_data.iloc[5000:]

#Split into x and y components
loan_training_data_x = loan_training_data.drop(columns=["default"])
loan_training_data_y = loan_training_data["default"]
loan_testing_data_x = loan_testing_data.drop(columns=["default"])
loan_testing_data_y = loan_testing_data["default"]

#Regression model
model = LogisticRegression(max_iter=1000)
model.fit(loan_training_data_x,loan_training_data_y)
y_predictions = model.predict(loan_testing_data_x)
results = abs(loan_testing_data_y-y_predictions)
results = results.values
regression_success_rate = np.count_nonzero(results == 0)/len(results)

#Decision Tree
model_2 = DecisionTreeClassifier(random_state= 10)
model_2.fit(loan_training_data_x,loan_training_data_y)
y_predictions = model_2.predict(loan_testing_data_x)
results = abs(loan_testing_data_y-y_predictions)
results = results.values
tree_success_rate = np.count_nonzero(results == 0)/len(results)


#Random forest
model_3 = RandomForestClassifier(random_state= 10)
model_3.fit(loan_training_data_x,loan_training_data_y)
y_predictions = model_3.predict(loan_testing_data_x)
results = abs(loan_testing_data_y-y_predictions)
results = results.values
forest_success_rate = np.count_nonzero(results == 0)/len(results)


#We opt for decision forest in our model
def expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score):
    row = pd.DataFrame({
        'credit_lines_outstanding': [credit_lines_outstanding],
        'loan_amt_outstanding': [loan_amt_outstanding],
        'total_debt_outstanding': [total_debt_outstanding],
        'income': [income],
        'years_employed': [years_employed],
        'fico_score': [fico_score]
    })
    default_probability = model_3.predict_proba(row)[:, 1]
    return default_probability[0]*0.9*loan_amt_outstanding


if __name__ == "__main__":
    print(regression_success_rate,tree_success_rate,forest_success_rate)
    print(expected_loss(4,3000,13067.57021,50352.16821,3,545))