# datasett_aarsregnskaper
This code is for master's theses I supervise.

## code_for_predictions_and_LASSO.py 
Produces some of the values of Table 2 in the following study:\
Paraschiv, F., Schmid, M., & Wahlstrøm, R. R. (2022). Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.\
Available at SSRN: https://dx.doi.org/10.2139/ssrn.3911490 \
(The code is not the same as used in the study, so some minor deviations from Table 2 will occur)

The code also creates a LASSO path for each training set, which is saved in the folder "results_plots"

## functions_predictions.py
Functions used in code_for_predictions.py 

## missing_values.py
This code exemplifies handling of imputation for completing missing values using k-Nearest Neighbors. Both continious and categorical variables are handled with this code. 