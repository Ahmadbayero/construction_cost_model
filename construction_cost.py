import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

###################################################################################################
### uncertainty factors meaning:
# A1 Weather 
# B2 Crew absenteeism 
# C3 Regulatory requirements (interpretation and implementation of government policy)
# D4 Design changes 
# E5 Scarcity of resources due to geographical location (economic activity level)
# F6 Social or political discontent of work men (labour unrest)
# G7 Crew interfacing 
# H8 Project complexity 
# I9 Ground/soil conditions (foundation conditions) 
# G10 Space congestion (overcrowding of workmen due to interface of activities under progress)

# K11 Managerial ability of consultant team involved 
# L12 Legal problems 
# M13 Rework due to poor material quality 
# N14 Rework due to poor work poor workmanship 
# O15 Many parties are involved directly 
# P16 Inconvenient site access 
# Q17 Limited construction area 
# R18 Delays in decision-making by project owner 
# S19 Postponement of project

# T20 Delays in payment 
# U21 Late site handover 
# V22 Late submission of nominated materials 
# W23 Late design works
# X24 Mistake in design
# Y25 Inappropriate design
# Z26 Low qualification and professional training of employees
# AA27 Late inspection by consultants
# AB28 Late issuing of approval documents
# AC29 Financial problems (limitations on provision of credit)

# AD30 Force majeure
# AE31 Corruption (political issues)
# AF32 Inflation rate
# AG33 Interest rate
# AH34 Exchange rate (availability and fluctuation in foreign exchange)
# AI35 Social and cultural conditions in the region
# AJ36 Unestimated work amounts in project’s estimate
# AK37 Unclear responsibility limits and no strict contractual obligations
# AL38 Customs and import restrictions and procedures
# AM39 Unnecessary interference by client

# AN40 Oil price
# AO41 Transportation prices
# AP42 Personal interest among consultants
# AQ43 Global economic recession
# AR44 End-users interferences
# AS45 Insecurity
###################################################################################################

def load_and_preprocess_data(filepath):
    """Load the dataset and preprocess."""
    data = pd.read_excel(filepath)
    data.drop('Unnamed: 6', axis=1, inplace=True)
    return data


def calculate_mape(actual, predictions):
    actual, predictions = np.array(actual), np.array(predictions)
    return np.mean(np.abs((actual - predictions) / actual)) * 100

def calculate_wmape(actual, predictions):
    actual, predictions = np.array(actual), np.array(predictions)
    return np.sum(np.abs(actual - predictions)) / np.sum(actual) * 100


def main():
    filepath = 'Dataset for Cost Prediction.xlsx'
    data = load_and_preprocess_data(filepath)
    
    # Define project scope columns and uncertainty variables
    project_scope_columns = ['Initial Estimated Duration (Months)', 'Gross Floor Area (M2)', 'Building Height (Metres)']
    uncertainty_columns = [col for col in data.columns if col.startswith('A') or col.startswith('B') or col.isalpha()]
    
    X_scope = data[project_scope_columns]
    X_uncertainty = data[uncertainty_columns]
    y = data['Initial Contract Sum']
    y_adjustment = data['Actual Contract Sum'] - data['Initial Contract Sum']
    
    # Splitting the dataset into training and testing sets once
    X_scope_train, X_scope_test, X_uncertainty_train, X_uncertainty_test, y_train, y_test, y_adj_train, y_adj_test = train_test_split(
        X_scope, X_uncertainty, y, y_adjustment, test_size=0.1, random_state=42)
    
    # Standardizing the project scope data
    scaler_scope = StandardScaler()
    X_scope_train_scaled = scaler_scope.fit_transform(X_scope_train)
    X_scope_test_scaled = scaler_scope.transform(X_scope_test)
    
    # Generating polynomial features for project scope variables
    poly = PolynomialFeatures(degree=3)
    X_scope_train_poly = poly.fit_transform(X_scope_train_scaled)
    X_scope_test_poly = poly.transform(X_scope_test_scaled)
    
    # Training the Ridge Regression model for initial contract sum prediction
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scope_train_poly, y_train)
    
    # Making initial predictions
    initial_predictions = ridge_model.predict(X_scope_test_poly)
    
    # Evaluating initial predictions
    initial_mape = mean_absolute_percentage_error(y_test, initial_predictions)
    print(f'Mean Absolute Percentage Error (MAPE) for Initial Contract Sum Predictions: {initial_mape*100:.2f}%')
    
    # Training the RandomForest model for cost adjustment prediction
    adjustment_model = RandomForestRegressor(n_estimators=150, criterion='absolute_error', random_state=42)
    adjustment_model.fit(X_uncertainty_train, y_adj_train)
    
    # Predicting adjustments
    adjustments_predicted = adjustment_model.predict(X_uncertainty_test)
    
    # Calculating final predictions
    final_predictions = initial_predictions + adjustments_predicted
    
    # Evaluating final predictions against actual contract sums
    actual_contract_sums_test = y_test + y_adj_test  # Corrected to use y_test and y_adj_test directly
    final_mape = mean_absolute_percentage_error(actual_contract_sums_test, final_predictions)
    print(f'Mean Absolute Percentage Error (MAPE) for Final Predictions: {final_mape*100:.2f}%')

if __name__ == '__main__':
    main()
