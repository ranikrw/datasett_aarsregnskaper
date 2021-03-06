##################################################################
##  Description                                                 ##
##################################################################
# This code produces some of the values of Table 2 in the following study:
# Paraschiv, F., Schmid, M., & Wahlstrøm, R. R. (2022). Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.
# Available at SSRN: https://dx.doi.org/10.2139/ssrn.3911490
# (The code is not the same as used in the study, so some minor deviations from Table 2 will occur)
##################################################################

import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import os

from functions_predictions_and_LASSO import *

##################################################################
##  Load data                                                   ##
##################################################################
# Select the folder with data.
# The folder should contain one file per accounting year, and nothing more
folder_name = '../../data5/'

print('-----------------------------------------')
print('Loading data:')
print('-----------------------------------------')
files = os.listdir(folder_name)
for current_file in files:
    file_year = int(current_file[0:4])

    # Loading one year file
    data_loaded = pd.read_csv(folder_name+current_file,sep=';',low_memory=False)

    # Adding all data together into data
    if current_file == files[0]:
        data = pd.DataFrame(columns=data_loaded.columns)
    data = pd.concat([data,data_loaded])
    print('Imported for accounting year {}'.format(file_year))

# Reset index 
data = data.reset_index(drop=True)

# Checking if all is unique
unique_orgnr = data.groupby(['orgnr','regnaar']).size().reset_index()
temp = unique_orgnr[0].unique()
if len(temp)==1:
    print('All orgnr unique')
else:
    print('ERROR: not all orgnr unique')

# Considering only accountint year 2019 or earlier
data = data[data.regnaar<=2019]
data = data.reset_index(drop=True) # Reset index

##################################################################
##  Preparing data                                              ##
##################################################################
# Considering only AS
data = data[data['orgform']=='AS']
data = data.reset_index(drop=True) # Reset index

# Excluding all small
# For many rows, accounting variables have no value. No value for accounting
# variables actually means that it is zero. Thus, use .fillna(0).
data = data[data['SUM EIENDELER'].fillna(0)>=5e5]
data = data.reset_index(drop=True) # Reset index 

# Considering only SMEs (https://ec.europa.eu/growth/smes/sme-definition_en)
ind = (data['sum_omsetning_EUR'].fillna(0)<=50e6) | (data['sum_eiendeler_EUR'].fillna(0)<=43e6)
data = data[ind]
data = data.reset_index(drop=True) # Reset index

# Excluding industries
data = data[data['naeringskoder_level_1']!='L'] # Real estate activities
data = data[data['naeringskoder_level_1']!='K'] # Financial and insurance activities
data = data[data['naeringskoder_level_1']!='D'] # Electricity and gas supply
data = data[data['naeringskoder_level_1']!='E'] # Water supply, sewerage, waste
data = data[data['naeringskoder_level_1']!='MISSING'] # Missing
data = data[data['naeringskoder_level_1']!='0'] # companies for investment and holding purposes only
data = data[data['naeringskoder_level_1']!='O'] # Public sector
data = data.reset_index(drop=True) # Reset index 

##################################################################
##  Make variables                                              ##
##################################################################
# Making variables for bankruptcy prediction
data,variable_list = make_variables_for_predictions(data)
# All variables are added to data as individual columns at the end
# variable_list is a list of all variables created

# As discussed in Paraschiv, Schmid & Wahlstrøm (2022), the variable
# "short-term liquidity / current assets" is not often selected, and
# also highly correlated with "inventory / current assets". Suggest
# therefore to remove it to avoid multicollinearity in the
# logistic regressions:
variable_list.remove('short-term liquidity / current assets')

##################################################################
##  Preprocessing data for predictions                          ##
##################################################################

interval_winsorizing = [0.01,0.99] # In numbers, so 0.01 = restricting at 1%
data = winsorizing_and_handling_inf(data,interval_winsorizing,variable_list)

# The functions make_variables_for_predictions and winsorizing_and_handling_inf also 
# do the following as described in Paraschiv, Schmid, & Wahlstrøm (2022):
 
# "If the denominator of a financial ratio is equal to zero, we set the value of 
# the variable to zero, if the numerator is equal to zero as well. If the numerator is 
# negative or positive, we set the ratio to the 1st or 99th percentiles of the distribution 
# of the ratio, respectively."

first_regnaar = 2010 # First accounting year to use for test
num_years_rolling_window = 4 # Number of accounting years to use for training data
data_folds = making_folds(first_regnaar,num_years_rolling_window,data)

##################################################################
##  Preparing modelling                                         ##
##################################################################
# Making index for result_fold
index_for_results = []
for name in variable_list:
    index_for_results.append(name)
index_for_results.append('intercept')
index_for_results.append('AUC on training set')
index_for_results.append('AUC on test set')
index_for_results.append('R-squared')
index_for_results.append('Number observations test set')
index_for_results.append('Number bankrupt in test set')

##################################################################
##  Parameters for the LASSO plots                              ##
##################################################################
# Set to True to show variable names next to lines on LASSO plot
do_show_labels = True

##################################################################
##  Making predictions                                          ##
##################################################################
import statsmodels.api as sm
from sklearn import metrics

# Empty data for results
results_table = pd.DataFrame(index=index_for_results)

# Empty data for full data set with predictions
data_with_predictions = pd.DataFrame(columns=data.columns)

for regnaar_test in data_folds.index:
    print('Modelling with test year {}'.format(regnaar_test))

    # Making training and test data
    training_data   = data_folds.loc[regnaar_test]['training_data']
    testing_data    = data_folds.loc[regnaar_test]['testing_data']

    # Shuffeling rows
    training_data   = training_data.sample(frac=1,random_state=0).reset_index(drop=True)
    testing_data    = testing_data.sample(frac=1,random_state=0).reset_index(drop=True)

    # Dividing between X (independent variables) and y (dependent variable)
    X_train, y_train, X_test, y_test = get_training_and_testing_data_for_fold(training_data,testing_data,variable_list)

    # Maximum iterations for training the LASSO and 
    # Logistic regression models
    maxiter = 1e6

    # Making and saving LASSO paths
    LASSO_path_rrw(X_train,y_train,maxiter,do_show_labels,regnaar_test)

    # Adding intercept
    X_train = sm.add_constant(X_train)
    X_test  = sm.add_constant(X_test)

    method = 'bfgs'
    model = sm.Logit(y_train, X_train)
    model = model.fit(maxiter=maxiter)
    
    # Printing statistics:
    # print(model.summary())

    # Making predictions
    y_hat_train  = model.predict(X_train)
    y_hat_test   = model.predict(X_test)

    # Making AUC on training data
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train,y_hat_train)
    auc_train = metrics.roc_auc_score(y_train,y_hat_train)

    # Making AUC on test data
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test,y_hat_test)
    auc_test = metrics.roc_auc_score(y_test,y_hat_test)

    # Making result_fold
    result_fold = pd.Series([None]*len(index_for_results), index=index_for_results, name=regnaar_test)
    result_fold['intercept'] = model.params[0]
    i = 0
    for name in variable_list:
        i +=1 
        result_fold[name] = model.params[i]
    result_fold['AUC on training set'] = auc_train
    result_fold['AUC on test set'] = auc_test
    result_fold['R-squared'] = model.prsquared
    result_fold['Number observations test set'] = len(y_test)
    result_fold['Number bankrupt in test set'] = sum(y_test)

    # Adding result_fold to table with all results
    results_table = results_table.assign(temp_name = result_fold)
    results_table.rename(columns={'temp_name':str(result_fold.name)}, inplace=True)

    # Making data with predictions
    testing_data['prediction'] = y_hat_test
    data_with_predictions = pd.concat([data_with_predictions,testing_data])
    data_with_predictions = data_with_predictions.reset_index(drop=True) # Reset index 

# Check data_with_predictions
# Checking that the sum of all variable values, across all observations, are the same
ind = (data['regnaar']<=np.max(data_folds.index)) & (data['regnaar']>=np.min(data_folds.index))
temp = data[ind]
temp1 = np.sum(np.sum(data_with_predictions[variable_list+['SUM EIENDELER']]))
temp2 = np.sum(np.sum(temp[variable_list+['SUM EIENDELER']]))
if temp1!=temp2:
    pst_change = np.abs((temp1-temp2)/temp1)
    if pst_change>1e-4: # Allow for up to 0.01% change due to computing errors
        print('ERROR: data_with_predictions not same as original data')

# Saving results to excel:
filename = 'results_table.xlsx'
results_table.to_excel(filename)
print('Results saved to '+filename)

# data_with_predictions is the original data set, but only for the years
# of which you have created test sets, and with predictions for each
# observation. You may want to save this data, or work further on it

