import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from sklearn.impute import KNNImputer

import time

import os

from sklearn.preprocessing import StandardScaler

##################################################################
##  Functions                                                   ##
##################################################################

def KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,n_neighbors_con,data,do_standardize_vector_space):
    # All variables that are imputed are treated as continious variables
    
    # var_mis_con: CONTINIOUS variables with missing values to be imputed
    # var_mis_cat: CATEGORICAL variables with missing values to be imputed
    # var_use: variables used in the vector space for the kNN algorithm for imputation 
    
    # n_neighbors_con: number of neighboring samples to use for imputation for continious variables
    # For categorical variables, n_neighbors is one for selecting the value of the one closest

    # Making sure that index is reset.
    # If not, the process will fail
    data = data.reset_index(drop=True) # Reset index

    # Checking that variables used for imputation do not have missing values
    temp = np.sum(pd.isnull(data[var_use]))
    temp = temp[temp!=0].index.tolist()
    if len(temp)>=1:
        for i in temp:
            print('-------------------------')
            print('ERROR: Variable \'{}\' is used for k-NN imputation. However, it has missing values. Missing values are set to zero in the k-NN imputation, so consider not including this variable.'.format(i))
        print('-------------------------')
        print('Imputing failed. No values imputed.')
        print('-------------------------')

    elif len(temp)==0:
        # Imputing variable, one at a time

        # Creating vector space used for the kNN-algorithm
        data_var_use = data[var_use]

        # Standardizing the vector space to a mean of 0 and 
        # standard deviation of 1 for each variable, respectively
        if do_standardize_vector_space:
            data_var_use = pd.DataFrame(StandardScaler().fit_transform(data_var_use),columns=data_var_use.columns)

        # First, continious variables
        for i in var_mis_con:
            t = time.time()
            print('-------------------------')

            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data[i],data_var_use],axis=1)
            num_missing_before = np.sum(pd.isnull(data[i]))

            # Imputing
            imputer = KNNImputer(n_neighbors=n_neighbors_con,weights='distance')
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data[i] = temp[0]

            num_missing_after = np.sum(pd.isnull(data[i]))
            print('Imputed {} instances of missing values for continious variable \'{}\''.format(num_missing_before-num_missing_after,i))
            print('Elapset time: {} minutes'.format(np.round((time.time()-t)/60,2)))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

        # Second, categorical variables including dummies
        for i in var_mis_cat:
            t = time.time()
            print('-------------------------')

            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data[i],data_var_use],axis=1)
            num_missing_before = np.sum(pd.isnull(data[i]))
            num_cat_values_before = len(data[i][pd.isnull(data[i])==False].unique())
            
            # Imputing
            imputer = KNNImputer(n_neighbors=1)
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data[i] = temp[0]

            num_missing_after = np.sum(pd.isnull(data[i]))
            num_cat_values_after = len(data[i][pd.isnull(data[i])==False].unique())

            print('Imputed {} instances of missing values for continious variable \'{}\''.format(num_missing_before-num_missing_after,i))
            print('Elapset time: {} minutes'.format(np.round((time.time()-t)/60,2)))
            if num_cat_values_before!=num_cat_values_after:
                print('ERROR: number of categorical values before and after impotation are {} and {}, respectively.'.format(num_cat_values_before,num_cat_values_after))
            else:
                print('Number of categorical values for variable \'{}\': {}'.format(i,num_cat_values_after))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

        print('-------------------------')

    return data

##################################################################
##  Simple example                                              ##
##################################################################
# Making simple data frame
data = pd.DataFrame([
    [1, 2, np.nan, 82.8, np.nan, 1, 2],
    [3, 4.8, 3, 7, 6, 0, 1],
    [np.nan, 6, 5.7, 1, 2, 0, 1],
    [8, 8, 7, 0.1, 22, 0, 3],
    [55, 3, 1, 3.4, 7, np.nan, 3],
    [1, 4.9, 8, 11, 2, 1, np.nan],
    [3, 44, 0, 1, 8, 1, 3]],
    columns=['a','b','c','d','e','f','g'])
# 'f' is a dummy variable, that is, it is categorical
# 'g' is categorical with three potential outcomes (1, 2, and 3)
# the other variables are continious

# Setting one value to None, just to illustrate 
# that None also is imputed
data.loc[2,'c'] = None

# number of neighboring samples to use for imputation for continious variables
n_neighbors_con = 3

# Set to True for standardizing the vector space for the kNN algorithm 
# to a mean of 0 and standard deviation of 1 for each variable, respectively
do_standardize_vector_space = True

# CONTINIOUS variables with missing values to be imputed
var_mis_con = [
    'a',
    'c',
]

# CATEGORICAL variables with missing values to be imputed
var_mis_cat = [
    'f',
    'g',
]

# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'b',
    'd',
    'e',
]
data = KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,n_neighbors_con,data,do_standardize_vector_space)
# This fails because 'e' has missing values

# Trying again only with variables fror the KNN without missing values: 
# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'b',
    'd',
]
data = KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,n_neighbors_con,data,do_standardize_vector_space)

# Note that for column 'e', there are still missing values

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
##  Selecting only some few (random) rows, as using all takes   ##
##  very long time to impute. After all, this code is           ##
##  only an example                                             ##
##################################################################
# Shuffle rows to make the extraction random
data = data.sample(frac=1).reset_index(drop=True)

# Extracting only 50 000 observations
data = data.iloc[0:50000]

##################################################################
##  Making a new continious variable with some missing values   ##
##################################################################
# Making a variable with random numbers between 0 and 1000
k = 1000
lenght = data.shape[0]
variable_with_missing_values = pd.Series(np.random.uniform(0,k,lenght))

# Making some missing values in it
variable_with_missing_values.at[3] = None
variable_with_missing_values.at[88] = None
variable_with_missing_values.at[75286] = None
variable_with_missing_values.at[1591565] = None
variable_with_missing_values.at[662535] = None
variable_with_missing_values.at[999565] = None
variable_with_missing_values.at[1209] = None
variable_with_missing_values.at[98565] = None
variable_with_missing_values.at[16495] = None
variable_with_missing_values.at[612865] = None

# Inserting this variable in data
data['new_var_con'] = variable_with_missing_values

##################################################################
##  Imputing the continious variable                            ##
##################################################################
# number of neighboring samples to use for imputation for continious variables
n_neighbors_con = 3

# Set to True for standardizing the vector space for the kNN algorithm 
# to a mean of 0 and standard deviation of 1 for each variable, respectively
do_standardize_vector_space = True

# CONTINIOUS variables with missing values to be imputed
var_mis_con = [
    'new_var_con',
]

# CATEGORICAL variables with missing values to be imputed
var_mis_cat = [
    'fravalg_revisjon', # This is a dummy with missing values
    'bistand_regnskapsforer', # This is a dummy with missing values
]

print('-------------------------')
print('Number of missing values for variable \'{}\' BEFORE imputing: {}'\
    .format(var_mis_con[0],np.sum(pd.isnull(data[var_mis_con[0]]))))
print('Number of missing values for variable \'{}\' BEFORE imputing: {}'\
    .format(var_mis_cat[0],np.sum(pd.isnull(data[var_mis_cat[0]]))))
print('Number of missing values for variable \'{}\' BEFORE imputing: {}'\
    .format(var_mis_cat[1],np.sum(pd.isnull(data[var_mis_cat[1]]))))
print('-------------------------')

# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'SUM EIENDELER',
    'can_opt_out',
    'age_in_days',
]
data = KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,n_neighbors_con,data,do_standardize_vector_space)

print('-------------------------')
print('Number of missing values for variable \'{}\' AFTER imputing: {}'\
    .format(var_mis_con[0],np.sum(pd.isnull(data[var_mis_con[0]]))))
print('Number of missing values for variable \'{}\' AFTER imputing: {}'\
    .format(var_mis_cat[0],np.sum(pd.isnull(data[var_mis_cat[0]]))))
print('Number of missing values for variable \'{}\' AFTER imputing: {}'\
    .format(var_mis_cat[1],np.sum(pd.isnull(data[var_mis_cat[1]]))))
print('-------------------------')