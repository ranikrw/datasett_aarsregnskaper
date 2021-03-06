import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from sklearn.svm import l1_min_c
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import lasso_path
# © 2007 - 2019, scikit-learn developers (BSD License).

import matplotlib.pyplot as plt

import os

import time

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
    data[name] = make_ratio(leverandorgjeld,sum_eiendeler)

    name = 'dummy; one if total liability exceeds total assets'
    variable_list.append(name)
    data[name] = sum_gjeld > sum_eiendeler

    name = '(current liabilities - short-term liquidity) / total assets'
    variable_list.append(name)
    temp = sum_omlopsmidler - bankinnskudd_kontanter_og_lignende
    data[name] = make_ratio(temp,sum_eiendeler)

    name = 'net income / total assets'
    variable_list.append(name)
    data[name] = make_ratio(arsresultat,sum_eiendeler)

    name = 'public taxes payable / total assets'
    variable_list.append(name)
    data[name] = make_ratio(skyldige_offentlige_avgifter,sum_eiendeler)

    name = 'interest expenses / total assets'
    variable_list.append(name)
    data[name] = make_ratio(annen_rentekostnad,sum_eiendeler)

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
    data[name] = make_ratio(sum_varer,sum_kortsiktig_gjeld)

    name = 'short-term liquidity / current assets'
    variable_list.append(name)
    data[name] = make_ratio(bankinnskudd_kontanter_og_lignende,sum_kortsiktig_gjeld)

    return data,variable_list


def make_ratio(numerator,denumerator):
    ratio = numerator.divide(denumerator)

    # If both numerator and denumerator are zero, the ratio is set to zero
    for i in range(len(ratio)):
        if numerator[i]==0:
            if denumerator[i]==0:
                ratio[i]=np.double(0)
    return ratio


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
    X_train = X_train.astype(float)
    y_train = y_train.astype(int)
        
    # Making test data (into ndarray)
    X_test = testing_data[variable_list]
    y_test = testing_data['bankrupt_fs']
    X_test = X_train.astype(float)
    y_test = y_train.astype(int)

    return X_train, y_train, X_test, y_test


def LASSO_path_rrw(X_train,y_train,maxiter,do_show_labels,regnaar_test):

        # Parameters that determine how the plots will look like
        fig_width  = 10 # Width of the figure
        fig_length = 10 # Length of the figure
        linewidth  = 2  # Width of the lines in the plots
        fontsize   = 18

        # Making folder for saving plots
        folder_name = 'results_plots'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Standardize the variables in the training set
        X_train_standardized = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)

        eps = 5e-3  # the smaller it is the longer is the path
        lambda_values, coefs_lasso, _ = lasso_path(X_train_standardized, y_train, eps=eps)

        # Plot LASSO path
        fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
        col_num = 0
        for coef_l in coefs_lasso:
            l1 = plt.plot(lambda_values, coef_l, linewidth=linewidth,label=X_train_standardized.columns[col_num])
            col_num +=1
        ax.set_xlim(ax.get_xlim()[::-1]) # Making descending x-axis
        if do_show_labels:
            textno=0
            y_values=[]
            for line in ax.lines:
                y = line.get_ydata()[-1]
                y_values.append(y)
                ax.annotate(X_train_standardized.columns[textno], xy=(.95,y), xytext=(3,0),xycoords = ax.get_yaxis_transform(), textcoords="offset points",size=fontsize, va="center")
                textno+=1
        ax.legend(loc = 'best', fontsize=fontsize,bbox_to_anchor=(1, -0.1),fancybox=False, shadow=False, ncol=1)
        ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
        ax.set_ylabel('Standardized coefficients',fontsize=fontsize)
        # ax.set_title('LASSO Path',fontsize=fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
        # plt.show() # Uncomment to show plot in Kernel
        plt.savefig(folder_name+'/LASSO_path_'+str(regnaar_test)+'.png',dpi=150, bbox_inches='tight')
        plt.close() # So the figures do not overlap


        ##################################################################
        ##  Alternative LASSO path                                      ##
        ##################################################################
        # Below is an alternative, if not better, way of making the LASSO path
        # However, I do not get the function l1_min_c to give correct value
        # Thus, I use the code above instead

        if False: # Set to False to skip all below
            # Defining a LASSO model
            model = linear_model.LogisticRegression(\
                solver='saga',
                penalty='l1',
                max_iter=maxiter,
                tol=1e-4,
                C=1, # This will be set to other values below
                fit_intercept=True,
                random_state=0)

            # Making C-values for the LASSO path
            # C is the inverse of lambda
            c_min = l1_min_c(X_train_standardized, y_train, loss='log')
            cs = c_min*np.logspace(0, 7, 20)

            # Making LASSO path and training AUC
            auc_train = []
            t=time.time()
            df_coefs = pd.DataFrame(index=X_train_standardized.columns)
            for c in cs:
                model.set_params(C=c) # C is the inverse of lambda
                model.fit(X_train_standardized, y_train)
                lambda_val = 1/c
                df_coefs[lambda_val] = pd.Series(model.coef_.ravel(),name=lambda_val,dtype=float,index=X_train_standardized.columns)
                auc_train.append(metrics.roc_auc_score(y_train,model.predict_proba(X_train_standardized)[:,1]))
            print('Elapset time total LASSO path: {} minutes'.format(np.round(((time.time()-t))/60,2)))

            # Transposing dataframe
            df_coefs = df_coefs.transpose()

            # Plot LASSO path
            fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
            ax.plot(df_coefs, linewidth=linewidth,label=df_coefs.columns)
            ax.set_xlim(ax.get_xlim()[::-1]) # Making descending x-axis
            if do_show_labels:
                textno=0
                y_values=[]
                for line in ax.lines:
                    y = line.get_ydata()[-1]
                    y_values.append(y)
                    ax.annotate(df_coefs.columns[textno], xy=(.95,y), xytext=(3,0),xycoords = ax.get_yaxis_transform(), textcoords="offset points",size=fontsize, va="center")
                    textno+=1
            ax.legend(loc = 'best', fontsize=fontsize,bbox_to_anchor=(1, -0.1),fancybox=False, shadow=False, ncol=1)
            ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
            ax.set_ylabel('Standardized coefficients',fontsize=fontsize)
            # ax.set_title('LASSO Path',fontsize=fontsize)
            ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
            # plt.show() # Uncomment to show plot in Kernel
            plt.savefig(folder_name+'/LASSO_path_'+str(regnaar_test)+'.png',dpi=150, bbox_inches='tight')
            plt.close() # So the figures do not overlap

            # Plot AUC scores for LASSO path
            fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
            ax.plot(df_coefs.index.tolist(), auc_train, linewidth=linewidth)
            ax.set_xlim(ax.get_xlim()[::-1]) # Making descending x-axis
            ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
            ax.set_ylabel('AUC on training set',fontsize=fontsize)
            ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
            # plt.show() # Uncomment to show plot in Kernel
            plt.savefig(folder_name+'/LASSO_path_AUC_scores'+'_'+str(regnaar_test)+'.png',dpi=150, bbox_inches='tight')
            plt.close() # So the figures do not overlap


