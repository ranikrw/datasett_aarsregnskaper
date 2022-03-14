import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def make_variables_for_predictions(data):

    # Retrieving relevant accounting values from the financial statements
    # For many rows, accounting variables have no value. No value for accounting
    # variables actually means that it is zero. Thus, use .fillna(0).

    bankinnskudd_kontanter_og_lignende      = make_variable_string1_but_if_not_string2(data,'Bankinnskudd, kontanter og lignende','Sum bankinnskudd, kontanter og lignende')

    skyldige_offentlige_avgifter            = data['Skyldige offentlige avgifter'].fillna(0)
    leverandorgjeld                         = data['Leverandoergjeld'].fillna(0)
    sum_kortsiktig_gjeld                    = data['Sum kortsiktig gjeld'].fillna(0)
    sum_innskutt_egenkapital                = data['Sum innskutt egenkapital'].fillna(0)
    sum_egenkapital                         = data['Sum egenkapital'].fillna(0)
    sum_eiendeler                           = data['SUM EIENDELER'].fillna(0)
    sum_gjeld                               = data['Sum gjeld'].fillna(0)
    arsresultat                             = data['Aarsresultat'].fillna(0)
    annen_rentekostnad                      = data['Sum finanskostnader'].fillna(0)
    sum_varer                               = data['Sum varer'].fillna(0)
    sum_omlopsmidler                        = data['Sum omloepsmidler'].fillna(0)


    age_in_days                             = data['age_in_days']

    # Not in use here, but uncomment if you want to use these:
    # nedskrivninger                          = data['Nedskrivning av varige driftsmidler og immaterielle eiendeler'].fillna(0)
    # kundefordringer                         = data['Kundefordringer'].fillna(0)
    # annen_renteinntekt                      = data['Annen renteinntekt'].fillna(0)
    # utbytte                                 = data['Utbytte'].fillna(0)
    # opptjent_egenkapital                    = data['Sum opptjent egenkapital'].fillna(0)
    # gjeld_til_kredittinstitusjoner          = data['Gjeld til kredittinstitusjoner'].fillna(0)
    # salgsinntekt                            = data['Salgsinntekt'].fillna(0)
    # lonnskostnad                            = data['Loennskostnad'].fillna(0)
    # avskrivninger                           = data['Avskrivning paa varige driftsmidler og immaterielle eiendeler'].fillna(0)
    # ordinaert_resultat_foer_skattekostnad   = data['Ordinaert resultat foer skattekostnad'].fillna(0)
    # ordinaert_resultat_etter_skattekostnad  = data['Ordinaert resultat etter skattekostnad'].fillna(0)
    # sum_inntekter                           = data['Sum inntekter'].fillna(0)


    ##################################################################
    ##  Making variables:                                           ##
    ##################################################################
    # Variables from the following study:
    # Paraschiv, F., Schmid, M., & Wahlstrøm, R. R. (2022). Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.
    # Available here: https://dx.doi.org/10.2139/ssrn.3911490

    variable_list = [] # This is a list of variables created, such that they easily can be used in models

    name = 'accounts payable / total assets'
    variable_list.append(name)
    ratio = leverandorgjeld.divide(sum_eiendeler)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,leverandorgjeld,sum_eiendeler)

    name = 'dummy; one if total liability exceeds total assets'
    variable_list.append(name)
    data[name] = sum_gjeld > sum_eiendeler

    name = '(current liabilities - short-term liquidity) / total assets'
    variable_list.append(name)
    temp = sum_omlopsmidler - bankinnskudd_kontanter_og_lignende
    ratio = temp.divide(sum_eiendeler)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,temp,sum_eiendeler)

    name = 'net income / total assets'
    variable_list.append(name)
    ratio = arsresultat.divide(sum_eiendeler)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,arsresultat,sum_eiendeler)

    name = 'public taxes payable / total assets'
    variable_list.append(name)
    ratio = skyldige_offentlige_avgifter.divide(sum_eiendeler)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,skyldige_offentlige_avgifter,sum_eiendeler)

    name = 'interest expenses / total assets'
    variable_list.append(name)
    ratio = annen_rentekostnad.divide(sum_eiendeler)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,annen_rentekostnad,sum_eiendeler)

    name = 'dummy; one if paid-in equity is less than total equity'
    variable_list.append(name)
    data[name] = sum_egenkapital < sum_innskutt_egenkapital

    name = 'log(age in years)'
    variable_list.append(name)
    # Handling negative age
    ind = age_in_days<=0
    print('{} observations have negative age. Age set to zero for these.'.format(np.sum(ind)))
    age_in_days.loc[ind] = 1
    # Handling missing age
    ind = pd.isnull(age_in_days)
    print('{} observations have missing age. Age set to zero for these.'.format(np.sum(ind)))
    age_in_days.loc[ind] = 1
    data[name] = np.log(data['age_in_days']/365)

    name = 'inventory / current assets'
    variable_list.append(name)
    ratio = sum_varer.divide(sum_kortsiktig_gjeld)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,sum_varer,sum_kortsiktig_gjeld)

    name = 'short-term liquidity / current assets'
    variable_list.append(name)
    ratio = bankinnskudd_kontanter_og_lignende.divide(sum_kortsiktig_gjeld)
    data[name] = set_to_zero_if_num_and_denum_is_zero(ratio,bankinnskudd_kontanter_og_lignende,sum_kortsiktig_gjeld)


    return data,variable_list


def set_to_zero_if_num_and_denum_is_zero(fraction,numerator,denumerator):
    # If both numerator and denumerator are zero, the ratio is set to zero
    for i in range(len(fraction)):
        if numerator[i]==0:
            if denumerator[i]==0:
                fraction[i]=np.double(0)
    return fraction


def make_variable_string1_but_if_not_string2(data,string1,string2):
    # Most of the time, string1 is correct value.
    # However, sometimes string1 has missing value while
    # string2 has the correct value
    temp = pd.DataFrame()
    temp[string1]  = data[string1]
    temp[string2]  = data[string2]
    result = pd.Series([None]*len(temp))
    for i in range(len(temp)):
        if pd.isnull(temp[string1].iloc[i])==False:
            result[i] = temp[string1].iloc[i]
        elif pd.isnull(temp[string2].iloc[i])==False:
            result[i] = temp[string2].iloc[i]
        else: # if both is 'None'
            result[i]=np.double(0)
    return pd.to_numeric(result, errors='coerce')



def winsorizing_and_handling_inf(data,interval_winsorizing,variable_list):
    ############################################
    # First, handling Inf and -Inf by setting them to max and min, respectively
    ############################################
    for name in variable_list:
        data[name] = handle_inf(data[name])

    # Checking that handling Inf and -Inf went well
    for name in variable_list:
        column = data[name]
        if ((column == np.inf).sum()!=0):
            print('ERROR: some observations of variable {} are still inf.'.format(name))
        if ((column == -np.inf).sum()!=0):
            print('ERROR: some observations of variable {} are still -inf.'.format(name))


    ############################################
    # Second, winsorizing
    ############################################
    # data[name].quantile([.001,.01,0.1,0.25,.5,0.75,0.9,0.99,0.999])

    for name in variable_list:
        if data[name].dtype!='bool': # Skipping dummies
            data[name] = winsorize_rrw(data[name],interval_winsorizing)

    # Checking that winsorizing went well
    for name in variable_list:
        temp = data[name]
        if temp.dtype!='bool': # Skipping dummies
            # First, checking lower quantile of winsorizing:
            temp1   = temp.quantile([interval_winsorizing[0]]).iloc[0]
            temp2   = temp.quantile([interval_winsorizing[0]/10]).iloc[0]
            if temp1!=temp2:
                pst_change = np.abs((temp1-temp2)/temp1)
                if pst_change>1e3: # Allow for up to 0.1% difference due to acceptable computational errors
                    print('ERROR: winsorizing failed for variable {}.'.format(name))
            # Second, checking upper quantile of winsorizing:
            temp1       = temp.quantile([interval_winsorizing[1]]).iloc[0]
            temp_upper  = interval_winsorizing[1]*(1+1-interval_winsorizing[1])
            temp2       = temp.quantile([temp_upper]).iloc[0]
            if temp1!=temp2:
                pst_change = np.abs((temp1-temp2)/temp1)
                if pst_change>1e3: # Allow for up to 0.1% difference due to acceptable computational errors
                    print('ERROR: winsorizing failed for variable {}.'.format(name))

    return data

def handle_inf(column):
    column = column.replace(np.inf,np.max(column[column != np.inf]))
    column = column.replace(-np.inf,np.max(column[column != -np.inf]))
    return column


def winsorize_rrw(column,interval_winsorizing):
    lower=column.quantile(interval_winsorizing[0])
    upper=column.quantile(interval_winsorizing[1])
    return column.apply(lambda x: winsorize_rrw_inner(x,lower,upper))
    # return data.clip(lower=data.quantile(interval_winsorizing[0]), upper=data.quantile(interval_winsorizing[1]))

def winsorize_rrw_inner(x,lower,upper):
    if x<=lower:
        return lower
    elif x>=upper:
        return upper
    else:
        return x


def making_folds(first_regnaar,num_years_rolling_window,data):
    # Making df with all observations per regnaar
    unique_regnaar = np.sort(data.regnaar.unique())
    all_observations_per_regnaar = pd.DataFrame(columns = ['data'],index=unique_regnaar)
    for regnaar in unique_regnaar:
        all_observations_per_regnaar.loc[regnaar] = {'data' : data[data['regnaar'] == regnaar]}

    # Making folds
    regnaar_for_test = unique_regnaar[unique_regnaar>=first_regnaar]
    data_folds = pd.DataFrame(columns = ['training_data','testing_data'],index=regnaar_for_test)
    for regnaar_test in regnaar_for_test:
        if (regnaar_test in unique_regnaar)==False:
            print('ERROR: Trying to use regnaar {} as testing data, but it does not exist'.format(regnaar_test))
        else:
            testing_data = all_observations_per_regnaar.loc[regnaar_test]['data']
            regnaar_for_train = range(regnaar_test-num_years_rolling_window,regnaar_test)
            for regnaar_train in regnaar_for_train:
                if (regnaar_train in unique_regnaar)==False:
                    print('ERROR: Trying to use regnaar {} in training data, but it does not exist'.format(regnaar_train))
                else:
                    if regnaar_train == np.min(regnaar_for_train):
                        training_data=all_observations_per_regnaar.loc[regnaar_train]['data']
                    else:
                        training_data = pd.concat([training_data,all_observations_per_regnaar.loc[regnaar_train]['data']])
        testing_data = testing_data.reset_index(drop=True) # Reset index
        training_data = training_data.reset_index(drop=True) # Reset index
        data_folds.loc[regnaar_test] = {'training_data':training_data,'testing_data':testing_data}

    # Checking
    for i in data_folds.index:
        testing_data            = data_folds.loc[i]['testing_data']
        training_data           = data_folds.loc[i]['training_data']
        unique_regnaar_test     = testing_data['regnaar'].unique()
        unique_regnaar_train    = training_data['regnaar'].unique()
        if len(unique_regnaar_test)!=1:
            print('ERROR')
        if len(unique_regnaar_train)!=num_years_rolling_window:
            print('ERROR')
        if np.sum(unique_regnaar_train>=unique_regnaar_test):
            print('ERROR: Data leakage')

    return data_folds


def get_training_and_testing_data_for_fold(training_data,testing_data,variable_list):
    # Dividing between X (independent variables) and y (dependent variable)
    
    if np.sum(np.sum(pd.isnull(training_data[variable_list])))!=0:
        print('ERROR: missing values in training data')
    if np.sum(np.sum(pd.isnull(training_data['bankrupt_fs'])))!=0:
        print('ERROR: missing values in training data')

    if np.sum(np.sum(pd.isnull(testing_data[variable_list])))!=0:
        print('ERROR: missing values in training data')
    if np.sum(np.sum(pd.isnull(testing_data['bankrupt_fs'])))!=0:
        print('ERROR: missing values in training data')

    # Making training data (into ndarray)
    X_train = training_data[variable_list]
    y_train = training_data['bankrupt_fs']
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_train = X_train.astype('float')
    y_train = y_train.astype('int')

    # Making test data (into ndarray)
    X_test = testing_data[variable_list]
    y_test = testing_data['bankrupt_fs']
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    X_test = X_test.astype('float')
    y_test = y_test.astype('int')

    return X_train, y_train, X_test, y_test