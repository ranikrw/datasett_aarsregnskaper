import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from sklearn.impute import KNNImputer

import os

##################################################################
##  Functions                                                   ##
##################################################################

def KNNImputer_rrw(var_mis,var_use,n_neighbors,data):
    # var_mis: variables with missing values to be imputed
    # var_use: variables used in the vector space for the kNN algorithm for imputation 
    # n_neighbors: number of neighboring samples to use for imputation

    # Checking that variables used for imputation do not have missing values
    temp = np.sum(pd.isnull(data[var_use]))
    temp = temp[temp!=0].index.tolist()
    if len(temp)>=1:
        for i in temp:
            print('-------------------------')
            print('ERROR: Variable \'{}\' is used for k-NN imputation. However, it has missing values. Missing values are set to zero in the k-NN imputation, so consider not including this variable.'.format(i))
        print('-------------------------')
        print('Imputing failed')

    if len(temp)==0:
        # Imputing variable, one at a time
        for i in var_mis:
            # Making data such that first column is the one to be imputed
            data_for_imputation = data[[i]+var_use]

            # Imputing
            imputer = KNNImputer(n_neighbors=n_neighbors,weights='distance')
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data[i] = temp[0]
    
    return data

##################################################################
##  Simple example                                              ##
##################################################################
# Making simple data frame
data = pd.DataFrame([
    [1, 2, np.nan, 82.8, np.nan],
    [3, 4.8, 3, 7, 6],
    [np.nan, 6, 5, 1, 2],
    [8, 8, 7, 0, 22]],
    columns=['a','b','c','d','e'])

# Setting one value to None
data.loc[2,'c'] = None

# number of neighboring samples to use for imputation
n_neighbors = 2


# variables with missing values to be imputed
var_mis = [
    'a',
    'c',
]

# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'b',
    'd',
    'e',
]
data = KNNImputer_rrw(var_mis,var_use,n_neighbors,data)
# This fails because 'e' has missing values

# Trying again only with variables fror the KNN without missing values: 
# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'b',
    'd',
]
data = KNNImputer_rrw(var_mis,var_use,n_neighbors,data)

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
##  Making a new variable with some missing values              ##
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
data['new_var'] = variable_with_missing_values

##################################################################
##  Imputing                                                    ##
##################################################################
print('Number of missing values before imputing: {}'.format(np.sum(pd.isnull(data['new_var']))))

# number of neighboring samples to use for imputation
n_neighbors = 2

# variables with missing values to be imputed
var_mis = [
    'new_var',
]

# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'SUM EIENDELER',
    'regnaar',
    'age_in_days',
]
data = KNNImputer_rrw(var_mis,var_use,n_neighbors,data)

print('Number of missing values after imputing: {}'.format(np.sum(pd.isnull(data['new_var']))))

