"""
Functions are used to forecaset House Price Index (HPI) using ARX model.

It includes two forecast performance evaluation test:
- Testing the equality of prediction mean squared errors: David I. Harvey, Stephen J. Leybourne, Paul Newbold (1997)
- Tests for Forecast Encompassing: David I. Harvey, Stephen J. Leybourne, Paul Newbold (1998)

"""

#%%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from IPython.display import display
from itertools import combinations
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression # ElasticNetCV,
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, KFold # cross_val_score, 
import scipy.stats as ss



#%%

def lags_list_function(lags, max_lag):
    """
    Generate a list of lag combinations based on the specified `lags` parameter and maximum lag value.

    Parameters
    ----------
    lags : {'Auto', 'glob', list of int}
        Determines the type of lag combinations to generate:
        - 'Auto': Automatically generates sequential lags up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Uses the specified list of lags directly.
    max_lag : int
        The maximum lag value to consider when generating lag combinations.

    Returns
    -------
    lags_list : list of list of int
        A list where each element is a list of integers representing a combination
        of lag values. The specific combinations depend on the input `lags` parameter.

    Examples
    --------
    >>> lags_list_function('Auto', 3)
    [[], [1], [1, 2], [1, 2, 3]]

    >>> lags_list_function('glob', 2)
    [[], [1], [2], [1, 2]]

    >>> lags_list_function([1, 2], 3)
    [[1, 2]]
    """
    full_lags = list(range(1, max_lag+1))
    all_combo = []
    for r in range(0, max_lag+1):
            for combo in combinations(full_lags, r):
                all_combo.append(list(combo))

    if lags == 'Auto':
        lags_list = [full_lags[:i] for i in range(0, max_lag+1)]
    elif lags == 'glob':
        lags_list = all_combo
    else:
        lags_list = [lags]
    
    return lags_list

#%%

def result_table(index, header, h_list, index_name, first_index='None'):
    """
    Create a multi-indexed DataFrame with specified index and column headers.

    Parameters
    ----------
    index : list of str
        A list of labels for the DataFrame's index.
    header : str
        The header label for the DataFrame's columns.
    h_list : list of int
        A list of integers that will be appended to the header label to create the column names.
    index_name : str
        The name to assign to the DataFrame's index.
    first_index : str, optional
        The label for the first index position. Defaults to 'None'.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with a MultiIndex for the columns, where the first level is the `header` and
        the second level corresponds to 'h=' followed by each element in `h_list`. The DataFrame's
        index is set to the provided `index` list, with an optional `first_index` prepended.

    Examples
    --------
    >>> result_table(['A', 'B', 'C'], 'Metric', [1, 2, 3], 'Category')
        Metric          
            h=1  h=2  h=3
    None   NaN   NaN   NaN
    A      NaN   NaN   NaN
    B      NaN   NaN   NaN
    C      NaN   NaN   NaN

    >>> result_table(['X', 'Y'], 'Value', [1, 2], 'Type', 'Start')
        Value      
            h=1  h=2
    Start   NaN   NaN
    X       NaN   NaN
    Y       NaN   NaN
    """
    df_index = pd.Index([first_index] + index, name=index_name)
    df_columns = pd.MultiIndex.from_tuples([(header, 'h='+str(i)) for i in h_list])
    df = pd.DataFrame(index=df_index, columns=df_columns)
    return df

#%%

def compute_ic_cv(y, X, metric, cv_criteria='MSFE', fit_intercept=True, cv=5, shuffle=False, n_iter=20, seed=None):
    """
    Compute the information criterion (IC) or cross-validation (CV) score based on the specified metric and criteria.

    Parameters
    ----------
    y : array-like or pandas.Series
        The dependent variable vector (target).
    X : array-like or pandas.DataFrame
        The independent variable matrix (features).
    metric : {'CV', 'IC'}
        The type of metric to compute:
        - 'CV': Cross-validation metric based on the `cv_criteria`.
        - 'IC': Information criterion (e.g., BIC).
    cv_criteria : {'MSFE', 'MAFE'}, optional
        The criterion used to compute the CV score when `metric` is 'CV':
        - 'MSFE': Mean Squared Forecast Error (negative mean squared error).
        - 'MAFE': Mean Absolute Forecast Error (negative mean absolute error).
        Default is 'MSFE'.
    fit_intercept : bool, optional
        Whether to calculate the intercept for the linear model. If set to False, no intercept will be used in calculations. Default is True.
    cv : int, optional
        The number of folds in cross-validation. Default is 5.
    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches in cross-validation. Default is False.
    n_iter : int, optional
        The number of iterations for cross-validation when `shuffle` is True. Default is 20.
    seed : int, optional
        The random seed for reproducibility when shuffling data in cross-validation. Default is None.

    Returns
    -------
    ic : float
        The computed information criterion (IC) or cross-validation (CV) score.

    Examples
    --------
    >>> compute_ic_cv(y, X, metric='CV', cv_criteria='MSFE', cv=5)
    0.045

    >>> compute_ic_cv(y, X, metric='IC')
    210.34
    """

    n_iter = n_iter if shuffle else 1

    if metric == 'CV':
        ics = []
        for i in range(n_iter):
            splitter = KFold(n_splits=cv, shuffle=shuffle, random_state=seed)
            scores = cross_validate(LinearRegression(fit_intercept=fit_intercept), X, y, cv=splitter, scoring=['neg_mean_absolute_error','neg_mean_squared_error'])
            # scores = cross_val_score(LinearRegression(fit_intercept=fit_intercept), X, y, cv=cv, scoring='neg_mean_squared_error')
                
            if cv_criteria == 'MAFE':
                ics.append(-scores['test_neg_mean_absolute_error'].mean())
            elif cv_criteria == 'MSFE':
                ics.append(-scores['test_neg_mean_squared_error'].mean())
                # ics.append(-scores.mean())
        
        ic = np.mean(ics)
        
    elif metric == 'IC':
        ic = sm.OLS(y, X).fit().info_criteria('bic') # .astype(float) should be added to X_train in CoCalc
    
    return ic
    
#%%

def lag_selector(df, lag_select, seed=None, cv_criteria='MSFE', cv=5, shuffle=False, n_iter=20, h=1, max_lag=13, exog=None, var_order='cross', y_lags='Auto', exog_lags='Auto', seasonal=False, verbose=0):
    """
    Select optimal lags for the dependent and exogenous variables using information criteria (IC) or cross-validation (CV) metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data. The first column is assumed to be the dependent variable ('y').
    lag_select : {'IC', 'CV'}
        The metric used to select the optimal lags:
        - 'IC': Information criterion (e.g., BIC).
        - 'CV': Cross-validation score based on `cv_criteria`.
    seed : int, optional
        Random seed for reproducibility in cross-validation. Default is None.
    cv_criteria : {'MSFE', 'MAFE'}, optional
        The criterion used for cross-validation when `lag_select` is 'CV':
        - 'MSFE': Mean Squared Forecast Error (default).
        - 'MAFE': Mean Absolute Forecast Error.
    cv : int, optional
        Number of folds for cross-validation. Default is 5.
    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches for cross-validation. Default is False.
    n_iter : int, optional
        Number of iterations for cross-validation when `shuffle` is True. Default is 20.
    h : int, optional
        Forecast horizon, indicating how many steps ahead the model is predicting. Default is 1.
    max_lag : int, optional
        The maximum lag order to consider for selection. Default is 13.
    exog : str, optional
        Name of the exogenous variable in the DataFrame. Default is None.
    var_order : {'cross', 'nested'}, optional
        The order in which lag selection is performed:
        - 'cross': Cross all combinations of lags for 'y' and 'exog'.
        - 'nested': Select the best lags for 'y' first, then choose the best lags for 'exog' based on the selected 'y' lags. Default is 'cross'.
    y_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the dependent variable:
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    exog_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the exogenous variable, if applicable:
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    seasonal : bool, optional
        Whether to include seasonal dummies (monthly) in the model. Default is False.
    verbose : int, optional
        If greater than 1, prints detailed information about the lag selection process. Default is 0.

    Returns
    -------
    y_lags : list of int
        The optimal lags for the dependent variable ('y') based on the specified metric.
    exog_lags : list of int
        The optimal lags for the exogenous variable based on the specified metric.
    IC : float
        The minimum information criterion (IC) or cross-validation score obtained.
    ics : dict
        A dictionary where keys are the IC/CV scores and values are the corresponding lags for 'y' and 'exog'.
        
    Examples
    --------
    >>> lag_selector_IC_CV(df, lag_select='CV', cv_criteria='MSFE', h=1)
    ([1, 2, 3], [1], 0.034, {...})

    >>> lag_selector_IC_CV(df, lag_select='IC', exog='ExogVar', h=2, max_lag=5)
    ([1, 3], [1], 210.45, {...})
    """
    df = df.copy()
    y_lags_list = lags_list_function(y_lags, max_lag)
    exog_lags_list = lags_list_function(exog_lags, max_lag) if exog else [[]]

    df['const'] = 1
    base_cols = ['const']
        
    for i in range(h, max_lag + h):
        df['y.L' + str(i)] = df.iloc[:,0].shift(i)
    if exog:
        for i in range(h, max_lag + h):
            df[exog + '.L' + str(i)] = df.loc[:,exog].shift(i)
    df = df.dropna() # To get the same results with ar_select_order(), we should drop NaNs before starting to get ICs. 
                     # That means we wont have, for example, the first 13 rows even when try to get IC of lag=[1,2] when max_lag=13 and h=1
                     
    if seasonal:
        dummies = pd.get_dummies(df.index.month, dtype='float').iloc[:, 1:]
        dummies.columns = [f's({i},12)' for i in dummies.columns]
        base_cols += [f's({i},12)' for i in dummies.columns]
        dummies.index = df.index
        df = df.reset_index().merge(dummies, left_on='Date', right_index=True).set_index('Date')
        
    ics = {}
    if var_order == 'cross':
        for y_lags in y_lags_list:
            y_lags_for_model = [i+h-1 for i in y_lags]
            for exog_lags in exog_lags_list:
                exog_lags_for_model = [i+h-1 for i in exog_lags]# if exog else []
                cols_label = base_cols + ['y.L' + str(i) for i in y_lags_for_model] + [exog + '.L' + str(i) for i in exog_lags_for_model]
                y = df['HPI']
                X = df[cols_label]
                ic = compute_ic_cv(y, X, metric=lag_select, cv_criteria=cv_criteria, fit_intercept=False, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
                ics[ic] = [y_lags, exog_lags]
        ics = {k: ics[k] for k in sorted(ics)}
        IC = min(ics.keys())
        y_lags = ics[IC][0]
        exog_lags = ics[IC][1]
        
    else:
        exog_lags = exog_lags if ((exog_lags != 'Auto') & (exog != None)) else []
        exog_lags_for_model = [l+h-1 for l in exog_lags]
        for y_lags in y_lags_list:
            y_lags_for_model = [i+h-1 for i in y_lags]
            cols_label = base_cols + ['y.L' + str(i) for i in y_lags_for_model] + [exog + '.L' + str(i) for i in exog_lags_for_model]
            y = df['HPI']
            X = df[cols_label]            
            ic = compute_ic_cv(y, X, metric=lag_select, cv_criteria=cv_criteria, fit_intercept=False, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
            ics[ic] = [y_lags, exog_lags]
        ics = {k: ics[k] for k in sorted(ics)}
        IC = min(ics.keys())
        y_lags = ics[IC][0]
        
        y_lags_for_model = [i+h-1 for i in y_lags]
        exog_lags_list.remove(exog_lags)
        for exog_lags in exog_lags_list:
            exog_lags_for_model = [i+h-1 for i in exog_lags]
            cols_label = base_cols + ['y.L' + str(i) for i in y_lags_for_model] + [exog + '.L' + str(i) for i in exog_lags_for_model]
            y = df['HPI']
            X = df[cols_label]            
            ic = compute_ic_cv(y, X, metric=lag_select, cv_criteria=cv_criteria, fit_intercept=False, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
            ics[ic] = [y_lags, exog_lags]
            
        ics = {k: ics[k] for k in sorted(ics)}
        IC = min(ics.keys())
        exog_lags = ics[IC][1]
        
    if verbose > 1:
        print(f'h={h}, exog: {exog}, ics: {ics}')
    
    return y_lags, exog_lags, IC, ics

#%%

def model(df, h=1, max_lag=3, exog=None, seasonal=False, lag_select='IC', seed=None, cv=5, shuffle=False, n_iter=20, y_lags='Auto', exog_lags='Auto', var_order='cross', train_cut=0.8, verbose=0, log=False, plot=False, original_hpi=None, original_scale=True):
    """
    Fit an autoregressive exogenous (ARX) model with selected lags using information criteria (IC) or cross-validation (CV) for dependent and independent variables, and evaluate its performance.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data. The first column is assumed to be the dependent variable ('y').
    h : int, optional
        Forecast horizon, indicating how many steps ahead the model is predicting. Default is 1.
    max_lag : int, optional
        The maximum lag order to consider for selection. Default is 3.
    exog : str, optional
        Name of the exogenous variable in the DataFrame. Default is None.
    seasonal : bool, optional
        Whether to include seasonal dummies (monthly) in the model. Default is False.
    lag_select : {'IC', 'CV'}, optional
        The metric used to select the optimal lags:
        - 'IC': Information criterion (e.g., BIC).
        - 'CV': Cross-validation score based on `cv_criteria`. Default is 'IC'.
    seed : int, optional
        Random seed for reproducibility in cross-validation. Default is None.
    cv : int, optional
        Number of folds for cross-validation. Default is 5.
    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches for cross-validation. Default is False.
    n_iter : int, optional
        Number of iterations for cross-validation when `shuffle` is True. Default is 20.
    y_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the dependent variable:
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    exog_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the exogenous variable, if applicable:
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    var_order : {'cross', 'nested'}, optional
        The order in which lag selection is performed:
        - 'cross': Cross all combinations of lags for 'y' and 'exog'.
        - 'nested': Select the best lags for 'y' first, then choose the best lags for 'exog' based on the selected 'y' lags. Default is 'cross'.
    train_cut : float or str or datetime-like, optional
        The cutoff point for splitting the data into training and validation sets. Can be a float between 0 and 1 representing the proportion of the data to use for training, or a specific index value. Default is 0.8.
    verbose : int, optional
        Level of verbosity for debugging or detailed output. Default is 0.
    log : bool, optional
        If True, the data is assumed to be log-transformed and predictions will be back-transformed. Default is False.
    plot : bool, optional
        If True, plots the actual vs. predicted values with a vertical line indicating the training/validation split. Default is False.
    original_hpi : pandas.Series, optional
        Original HPI (Housing Price Index) values for back-transforming predictions. Only used if `original_scale` is True. Default is None.
    original_scale : bool, optional
        Whether to return the predictions on the original scale (before any transformations). Default is True.

    Returns
    -------
    MAFE_val : float
        Mean Absolute Forecast Error on the validation set.
    MSFE_val : float
        Mean Squared Forecast Error on the validation set.
    MAFE_train : float
        Mean Absolute Forecast Error on the training set.
    MSFE_train : float
        Mean Squared Forecast Error on the training set.
    lags_set : dict
        Dictionary containing the selected lags for 'y' and 'exog'.
    IC : float
        The minimum information criterion (IC) or cross-validation score obtained during lag selection.
    res_full : pandas.Series
        The full set of predictions for the dependent variable, including both training and validation periods.
        
    Examples
    --------
    >>> MAFE_val, MSFE_val, MAFE_train, MSFE_train, lags_set, IC, res_full = model_IC_CV(df, h=1, max_lag=3, exog='ExogVar')
    >>> print(MAFE_val, MSFE_val, lags_set)
    0.034 0.002 {'y lags': [1, 2], 'exog lags': [1]}
    """
    df = df.copy()
    res_full = pd.Series(index=df.index)

    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]
    
    y_lags, exog_lags, IC, ics = lag_selector(df.loc[:train_cut], h=h, max_lag=max_lag, exog=exog, seasonal=seasonal, lag_select=lag_select, seed=seed, cv=cv, shuffle=shuffle, n_iter=n_iter, y_lags=y_lags, exog_lags=exog_lags, var_order=var_order, verbose=verbose)
    y_lags_for_model = [i+h-1 for i in y_lags]
    exog_lags_for_model = [i+h-1 for i in exog_lags] if exog else []
        
    df['const'] = 1
    base_cols = ['const']
    
    for i in range(h, max_lag + h):
        df['y.L' + str(i)] = df.iloc[:,0].shift(i)
    if exog:
        for i in range(h, max_lag + h):
            df[exog + '.L' + str(i)] = df.loc[:,exog].shift(i)
    df = df.dropna() # To get the same results with ar_select_order(), we should drop NaNs before starting to get ICs. 
                     # That means we wont have, for example, the first 13 rows even when try to get IC of lag=[1,2] when max_lag=13 and h=1
                     
    if seasonal:
        dummies = pd.get_dummies(df.index.month, dtype='float').iloc[:, 1:]
        dummies.columns = [f's({i},12)' for i in dummies.columns]
        base_cols += [f's({i},12)' for i in dummies.columns]
        dummies.index = df.index
        df = df.reset_index().merge(dummies, left_on='Date', right_index=True).set_index('Date')
        
    cols_label = base_cols + [f'y.L{lag}' for lag in y_lags_for_model] + [f'{exog}.L{lag}' for lag in exog_lags_for_model]
        
    Y = df['HPI']
    Y_train = Y.loc[:train_cut]
    X = df[cols_label]
    X_train = X.loc[:train_cut]
    
    model = sm.OLS(Y_train, X_train).fit() # .astype(float) should be added to X_train in CoCalc
    predict = model.predict(X)
    res_full.loc[:] = predict # res_full.update(predict) # res_full.combine_first(predict) # predict.reindex(res_full.index)

    if original_scale:
        if original_hpi is None:
            if log:
                res_full = np.exp(res_full)
                Y_full = np.exp(Y)
            else:
                Y_full = Y
        else:
            Y_full = original_hpi
            if log:
                res_full = np.exp(res_full + np.log(original_hpi) - Y)
            else:
                res_full = res_full + original_hpi - Y
    else:
        Y_full = Y
    
    # val_start_pos = df.index.searchsorted(train_cut, side='right')
    val_start_idx = df.index[df.index.get_loc(train_cut) + 1]
    e = Y_full - res_full
    e_train = e.loc[:train_cut]
    e_val = e.loc[val_start_idx:]
    MAFE_train = e_train.abs().mean()
    MAFE_val = e_val.abs().mean()
    MSFE_train = e_val.abs().mean()
    MSFE_val  = (e_val ** 2).mean()
    
    lags_set = {'y lags': y_lags, 'exog lags': exog_lags}
    
    if plot:
        plt.plot(res_full)
        plt.axvline(x = pd.Timestamp(train_cut), color = 'r', ls='--', label = 'train cut')
        plt.plot(Y_full)
    
    return MAFE_val, MSFE_val, MAFE_train, MSFE_train, lags_set, IC, res_full #, e_train, params

#%%

def compare_exog(df, train_cut, h_list=[1,3,6,12], seasonal=False, max_lag=3, lag_select='IC', seed=None, lag_fit_intercept=True, lag_cv=5, lag_shuffle=False, lag_iter=20, hsi_CV_select=False, hsi_fit_intercept=True, hsi_cv=4, hsi_iter=40, y_lags='Auto', exog_lags='Auto', var_order='cross', verbose=False, log=False, original_hpi=None, original_scale=True, sort_df='criteria',  sort_col=-1):
    """
    Compare the impact of different exogenous variables on forecasting accuracy using various evaluation metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the time series data. The first column is assumed to be the dependent variable ('HPI'), and the subsequent columns are potential exogenous variables.
    train_cut : float or str or datetime-like
        The cutoff point for splitting the data into training and validation sets. Can be a float between 0 and 1 representing the proportion of the data to use for training, or a specific index value.
    h_list : list of int, optional
        A list of forecast horizons (e.g., [1, 3, 6, 12]) to evaluate. Default is [1, 3, 6, 12].
    seasonal : bool, optional
        Whether to include seasonal dummies (monthly) in the model. Default is False.
    max_lag : int, optional
        The maximum lag order to consider for selection. Default is 3.
    lag_select : {'IC', 'CV'}, optional
        The metric used to select the optimal lags:
        - 'IC': Information criterion (e.g., BIC).
        - 'CV': Cross-validation score. Default is 'IC'.
    seed : int, optional
        Random seed for reproducibility in cross-validation. Default is None.
    lag_fit_intercept : bool, optional
        Whether to include an intercept in the lag selection linear regression model. Default is True.
    lag_cv : int, optional
        Number of folds for cross-validation during lag selection. Default is 5.
    lag_shuffle : bool, optional
        Whether to shuffle the data before splitting into batches for cross-validation during lag selection. Default is False.
    lag_iter : int, optional
        Number of iterations for cross-validation during lag selection when `lag_shuffle` is True. Default is 20.
    hsi_CV_select : bool, optional
        Whether to perform cross-validation for the selection of the UHSI (Unified Housing Sentiment Index) based on the selected lags. Default is False.
    hsi_fit_intercept : bool, optional
        Whether to include an intercept in the UHSI selection linear regression model. Default is True.
    hsi_cv : int, optional
        Number of folds for cross-validation during UHSI selection. Default is 4.
    hsi_iter : int, optional
        Number of iterations for cross-validation during UHSI selection when `hsi_CV_select` is True. Default is 40.
    y_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the dependent variable ('HPI'):
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    exog_lags : {'Auto', 'glob', list of int}, optional
        The lags to consider for the exogenous variables, if applicable:
        - 'Auto': Automatically generate lag sequences up to `max_lag`.
        - 'glob': Generates all possible combinations of lags up to `max_lag`.
        - list of int: Use the specified lags directly. Default is 'Auto'.
    var_order : {'cross', 'nested'}, optional
        The order in which lag selection is performed:
        - 'cross': Cross all combinations of lags for 'y' and 'exog'.
        - 'nested': Select the best lags for 'y' first, then choose the best lags for 'exog' based on the selected 'y' lags. Default is 'cross'.
    verbose : bool, optional
        If True, print detailed output during the function execution. Default is False.
    log : bool, optional
        If True, the data is assumed to be log-transformed and predictions will be back-transformed. Default is False.
    original_hpi : pandas.Series, optional
        Original HPI (Housing Price Index) values for back-transforming predictions. Only used if `original_scale` is True. Default is None.
    original_scale : bool, optional
        Whether to return the predictions on the original scale (before any transformations). Default is True.
    sort_df : {'criteria', 'MAFE', 'MSFE'}, optional
        Criteria to sort the results by:
        - 'criteria': Sort by the lag selection criteria (e.g., IC or CV).
        - 'MAFE': Sort by Mean Absolute Forecast Error.
        - 'MSFE': Sort by Mean Squared Forecast Error. Default is 'criteria'.
    sort_col : int, optional
        The column index in `sort_df` to use for sorting. Default is -1 (last column).

    Returns
    -------
    MAFE_val_df : pandas.DataFrame
        DataFrame containing the Mean Absolute Forecast Error (MAFE) on the validation set for each exogenous variable and forecast horizon.
    MAFE_val_df_improve : pandas.DataFrame
        DataFrame containing the percentage improvement in MAFE on the validation set for each exogenous variable compared to the baseline (no exogenous variables).
    MSFE_val_df : pandas.DataFrame
        DataFrame containing the Mean Squared Forecast Error (MSFE) on the validation set for each exogenous variable and forecast horizon.
    MSFE_val_df_improve : pandas.DataFrame
        DataFrame containing the percentage improvement in MSFE on the validation set for each exogenous variable compared to the baseline.
    criteria_df : pandas.DataFrame
        DataFrame containing the lag selection criteria (e.g., IC or CV) values for each exogenous variable and forecast horizon.
    lags_df : pandas.DataFrame
        DataFrame containing the selected lags for the dependent variable ('HPI') and each exogenous variable.
    MAFE_train_df : pandas.DataFrame
        DataFrame containing the Mean Absolute Forecast Error (MAFE) on the training set for each exogenous variable and forecast horizon.
    MSFE_train_df : pandas.DataFrame
        DataFrame containing the Mean Squared Forecast Error (MSFE) on the training set for each exogenous variable and forecast horizon.
    forecast_dict : dict
        A dictionary where each key is a forecast horizon (from `h_list`) and the corresponding value is a DataFrame containing the actual vs. predicted values for each exogenous variable.
    df : pandas.DataFrame
        The modified input DataFrame with additional columns for principal components (UHSI) if `hsi_CV_select` is True.
    
    Examples
    --------
    >>> MAFE_val_df, MAFE_val_df_improve, MSFE_val_df, MSFE_val_df_improve, criteria_df, lags_df, MAFE_train_df, MSFE_train_df, forecast_dict, df = compare_exog(df, '2023-01-01', h_list=[1, 6, 12], lag_select='IC', seasonal=True)
    >>> print(MAFE_val_df)
    Horizon 1  Horizon 6  Horizon 12
    exog_var1    0.03      0.04       0.05
    exog_var2    0.02      0.03       0.04
    """
    df = df.copy()
    n_hsi = df.shape[1] - 1
    index = list(df.columns[1:]) + ['UHSI_'+str(i) for i in h_list]
    
    if (original_hpi is None) | (original_scale == False):
        first_col = df.loc[:,'HPI']
    else:
        first_col = original_hpi
        
    forecast_df = pd.DataFrame(index=first_col.index, columns=['HPI', 'None']+index)
    # forecast_df.columns = forecast_df.columns[0] + ['Y_' + i for i in forecast_df.columns[1:]]
    forecast_df.iloc[:,0] = first_col
    forecast_dict = {h:forecast_df.copy() for h in h_list}
    
    MAFE_val_df = result_table(index, 'MAFE_val', h_list, 'exog')
    MAFE_val_df_improve = result_table(index, 'MAFE_val improve (in percent)', h_list, 'exog')
    MSFE_val_df = result_table(index, 'MSFE_val', h_list, 'exog')
    MSFE_val_df_improve = result_table(index, 'MSFE_val improve (in percent)', h_list, 'exog')
    MAFE_train_df = result_table(index, 'MAFE_train', h_list, 'exog')
    MSFE_train_df = result_table(index, 'MSFE_train', h_list, 'exog')
    lags_df = result_table(index, 'lags', h_list, 'exog')
    criteria_df = result_table(index, 'lag selection criteria: ' + lag_select, h_list, 'exog')
    if hsi_CV_select & (lag_select in ['IC', 'CV']):
        criteria_df = result_table(index, 'hsi selection criteria: CV', h_list, 'exog')
    
    for i, exog in enumerate([None]+index):
        for j, h in enumerate(h_list):
                        
            if exog == f'UHSI_{h_list[0]}':
                top_hsis = criteria_df.iloc[1:n_hsi+1].sort_values(criteria_df.columns[j]).index[:10]
                if verbose:
                    print(f'10 Selected queries for UHSI_{h}:\n{np.array(top_hsis)}')
                hsi_selected = df.loc[:, top_hsis]
                hsi_selected_train = hsi_selected.loc[:train_cut]
                pca = PCA(n_components=1)
                principal_component = pca.fit(hsi_selected_train).transform(hsi_selected)
                df['UHSI_'+str(h)] = principal_component
            
            MAFE_val, MSFE_val, MAFE_train, MSFE_train, lag_set, criteria, Y_pred = model(df, h=h, exog=exog, seasonal=seasonal, max_lag=max_lag, lag_select=lag_select, seed=seed, fit_intercept=lag_fit_intercept, cv=lag_cv, shuffle=lag_shuffle, n_iter=lag_iter, y_lags=y_lags, exog_lags=exog_lags, var_order=var_order, train_cut=train_cut, verbose=verbose, log=log, plot=False, original_hpi=original_hpi, original_scale=original_scale)
            if hsi_CV_select:
                MAFE_val, MSFE_val, MAFE_train, MSFE_train, lag_set, criteria, Y_pred = model(df, h=h, exog=exog, seasonal=seasonal, max_lag=max_lag, lag_select='CV', seed=seed, fit_intercept=hsi_fit_intercept, cv=hsi_cv, shuffle=True, n_iter=hsi_iter, y_lags=lag_set['y lags'], exog_lags=lag_set['exog lags'], var_order=var_order, train_cut=train_cut, verbose=verbose, log=log, plot=False, original_hpi=original_hpi, original_scale=original_scale)
                    
            criteria_df.iloc[i, j] = criteria
            MAFE_val_df.iloc[i, j] = MAFE_val
            MAFE_val_df_improve.iloc[i, j] = 100 * (MAFE_val_df.iloc[0, j] - MAFE_val) / MAFE_val_df.iloc[0, j]
            MSFE_val_df.iloc[i, j] = MSFE_val
            MSFE_val_df_improve.iloc[i, j] = 100 * (MSFE_val_df.iloc[0, j] - MSFE_val) / MSFE_val_df.iloc[0, j]
            MAFE_train_df.iloc[i, j] = MAFE_train
            MSFE_train_df.iloc[i, j] = MSFE_train
            lags_df.iloc[i, j] = str(lag_set)
            forecast_dict[h].iloc[-len(Y_pred):, i+1] = Y_pred
    
    MAFE_val_df_improve.loc['selected'] = [MAFE_val_df_improve.iloc[criteria_df.iloc[:,i].argmin(), i] for i in range(criteria_df.shape[1])]
    MSFE_val_df_improve.loc['selected'] = [MSFE_val_df_improve.iloc[criteria_df.iloc[:,i].argmin(), i] for i in range(criteria_df.shape[1])]
    
    # Sorting dfs
    if sort_df == 'MAFE':        
        MAFE_val_df = MAFE_val_df.loc[['None']+list(MAFE_val_df.iloc[1:,:].sort_values(MAFE_val_df.columns[sort_col]).index)]
    elif sort_df == 'MSFE':        
        MSFE_val_df = MSFE_val_df.loc[['None']+list(MSFE_val_df.iloc[1:,:].sort_values(MSFE_val_df.columns[sort_col]).index)]
        MAFE_val_df = MAFE_val_df.loc[MSFE_val_df.index]
    elif sort_df == 'criteria':
        criteria_df = criteria_df.loc[['None']+list(criteria_df.iloc[1:,:].sort_values(criteria_df.columns[sort_col]).index)]
        MAFE_val_df = MAFE_val_df.loc[criteria_df.index]
        
    MAFE_val_df_improve = MAFE_val_df_improve.loc[['selected'] + list(MAFE_val_df.index[1:])]
    MSFE_val_df = MSFE_val_df.loc[MAFE_val_df.index]
    MSFE_val_df_improve = MSFE_val_df_improve.loc[['selected'] + list(MAFE_val_df.index[1:])]
    criteria_df = criteria_df.loc[MAFE_val_df.index]
    MAFE_train_df = MAFE_train_df.loc[MAFE_val_df.index]
    MSFE_train_df = MSFE_train_df.loc[MAFE_val_df.index]
    lags_df = lags_df.loc[MAFE_val_df.index]

    return MAFE_val_df, MAFE_val_df_improve, MSFE_val_df, MSFE_val_df_improve, criteria_df, lags_df, MAFE_train_df, MSFE_train_df, forecast_dict, df

#%%

def HLN_MDM(d, h):
    """
    Compute the Harvey, Leybourne, and Newbold (1997) Modified Diebold-Mariano (MDM) statistic and its p-value.

    Parameters
    ----------
    d : numpy.ndarray or pandas.Series
        The array or series of forecast error differentials. This represents the difference between the forecast errors of two competing models.
    h : int
        The forecast horizon (number of steps ahead).

    Returns
    -------
    MDM : float
        The Modified Diebold-Mariano statistic.
    pval : float
        The p-value associated with the MDM statistic, under the null hypothesis that the forecast accuracy of the two models is the same.

    Notes
    -----
    - The Harvey, Leybourne, and Newbold (1997) test modifies the Diebold-Mariano test to account for small-sample bias, especially when the forecast horizon is greater than one.
    - This test is particularly useful in comparing predictive accuracy when forecasts are based on overlapping data.

    References
    ----------
    Harvey, D. I., Leybourne, S. J., & Newbold, P. (1997). 
    Testing the equality of prediction mean squared errors. 
    International Journal of Forecasting, 13(2), 281-291.

    Examples
    --------
    >>> d = np.array([0.1, 0.2, -0.1, 0.05, 0.3])
    >>> h = 1
    >>> MDM, pval = HLN_MDM(d, h)
    >>> print(f"MDM Statistic: {MDM}, p-value: {pval}")
    MDM Statistic: 1.414213562373095, p-value: 0.1826592511440174
    """
    
    tau = d.shape[0]
    MDMfix = np.sqrt((tau +1 -2*h + h*(h-1)/tau) / tau )
    d_bar = np.mean(d)
    sigma_d = np.sum(d*d)
    if h > 1:
        for j in range(h-1):
            sigma_d += 2 * np.sum(d[1+j:]*d[:tau-j-1])
    sigma_d = np.sqrt(sigma_d) / tau
    MDM = MDMfix * d_bar/sigma_d
    pval = 2*ss.t.cdf(-abs(MDM),tau-1)
    
    return MDM, pval

#%%

def forecast_table(self):
    """
    Generate a table comparing forecast performance across different predictors and horizons.

    This table includes the Mean Absolute Forecast Error (MAFE), Mean Squared Forecast Error (MSFE),
    and p-values for hypothesis tests related to equal predictive accuracy and forecast encompassing.

    Hypotheses:
    -----------
    - H_{0,1}: Equal MAFE between the predictor model and the base model.
    - H_{0,2}: Equal MSFE between the predictor model and the base model.
    - H_{0,3}: The predictor model forecast encompasses the base model.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame with a MultiIndex of forecast horizons (`h`) and predictors, and columns including:
        - 'HPI lags': The lags of the dependent variable (HPI).
        - 'Exog lags': The lags of the exogenous variable.
        - 'MAFE': The Mean Absolute Forecast Error for the predictor.
        - 'MSFE': The Mean Squared Forecast Error for the predictor.
        - 'MAFE improvement': The percentage improvement in MAFE relative to the base model.
        - 'MSFE improvement': The percentage improvement in MSFE relative to the base model.
        - 'H_{0,1}': The p-value for the hypothesis test of equal MAFE (using the Modified Diebold-Mariano test).
        - 'H_{0,2}': The p-value for the hypothesis test of equal MSFE (using the Modified Diebold-Mariano test).
        - 'H_{0,3}': The p-value for the hypothesis test of forecast encompassing (using the Modified Diebold-Mariano test).

    Notes
    -----
    - The base model is represented by the 'None' predictor.
    - Forecast encompassing tests whether the predictor model contains all the information in the base model.
    - The method uses the Harvey, Leybourne, and Newbold (1997) Modified Diebold-Mariano test to compute p-values.

    Examples
    --------
    >>> table = model.forecast_table()
    >>> print(table)
                    HPI lags Exog lags     MAFE     MSFE MAFE improvement MSFE improvement   H_{0,1}   H_{0,2}   H_{0,3}
    h    Predictors                                                                                                      
    1    None              ...       ...  0.0251  0.00123            0.000            0.000  0.0321    0.0287    0.1025
        UHSI_1            ...       ...  0.0223  0.00110            0.111            0.105  0.2103    0.1325    0.0923
        UHSI_3            ...       ...  0.0210  0.00105            0.162            0.147  0.1809    0.1014    0.0534
    ...
    """
    Predictors = ['None', 'UHSI_1', 'UHSI_3', 'UHSI_6', 'UHSI_12']
    result = pd.DataFrame(index=pd.MultiIndex.from_product([self.h_list, Predictors], names=['h', 'Predictors']), 
                          columns=['HPI lags', 'Exog lags', 'MAFE', 'MSFE', 'MAFE improvement', 'MSFE improvement', 'H_{0,1}', 'H_{0,2}', 'H_{0,3}'])
    
    for h in self.h_list:
        df = self.forecast_dict[h]
        e_base = (df['HPI'] - df['None'])[self.train_cut:].iloc[1:]
        for exog in Predictors:
            lags_set = ast.literal_eval(self.lags_df.loc[exog, ('lags', f'h={h}')])
            result.loc[(h, exog), 'HPI lags'] = str(lags_set['y lags'])
            result.loc[(h, exog), 'Exog lags'] = str(lags_set['exog lags'])
            result.loc[(h, exog), 'MAFE'] = self.MAFE_val_df.loc[exog, ('MAFE_val', f'h={h}')]
            result.loc[(h, exog), 'MSFE'] = self.MSFE_val_df.loc[exog, ('MSFE_val', f'h={h}')]
            result.loc[(h, exog), 'MAFE improvement'] = (1 - result.loc[(h, exog), 'MAFE']/result.loc[(h, 'None'), 'MAFE'])
            result.loc[(h, exog), 'MSFE improvement'] = (1 - result.loc[(h, exog), 'MSFE']/result.loc[(h, 'None'), 'MSFE'])
            e_exog = (df['HPI'] - df[exog])[self.train_cut:].iloc[1:]
            d = e_exog.abs() - e_base.abs()
            MDM, pval = HLN_MDM(d,h)
            result.loc[(h, exog), 'H_{0,1}'] = pval
            d = e_exog**2 - e_base**2
            MDM, pval = HLN_MDM(d,h)
            result.loc[(h, exog), 'H_{0,2}'] = pval
            d = (e_exog - e_base) * e_exog
            MDM, pval = HLN_MDM(d,h)
            result.loc[(h, exog), 'H_{0,3}'] = pval
        
    return result

#%%

def results(self, results_table=False, MAFE_val_df=False, MAFE_val_df_improve=False, MSFE_val_df=False, MSFE_val_df_improve=False, forecast_criteria_df=False, lags_df=False, MAFE_train_df=False, MSFE_train_df=False, head=10):
    """
    Display selected result tables generated during the forecasting process.

    Parameters
    ----------
    results_table : bool, optional
        If True, display the full results table from the forecast comparison (default is False).
    MAFE_val_df : bool, optional
        If True, display the Mean Absolute Forecast Error (MAFE) validation DataFrame (default is False).
    MAFE_val_df_improve : bool, optional
        If True, display the percentage improvement in MAFE for each predictor (default is False).
    MSFE_val_df : bool, optional
        If True, display the Mean Squared Forecast Error (MSFE) validation DataFrame (default is False).
    MSFE_val_df_improve : bool, optional
        If True, display the percentage improvement in MSFE for each predictor (default is False).
    forecast_criteria_df : bool, optional
        If True, display the forecast criteria DataFrame, showing the selected model criteria (default is False).
    lags_df : bool, optional
        If True, display the DataFrame containing the selected lags for each model (default is False).
    MAFE_train_df : bool, optional
        If True, display the Mean Absolute Forecast Error (MAFE) training DataFrame (default is False).
    MSFE_train_df : bool, optional
        If True, display the Mean Squared Forecast Error (MSFE) training DataFrame (default is False).
    head : int or bool, optional
        Number of rows to display from each DataFrame (default is 10). 
        If True, display all rows of each DataFrame.

    Returns
    -------
    None
        Displays the selected DataFrames in the Jupyter Notebook environment.

    Notes
    -----
    - The method allows selective display of any of the key result tables generated during the model comparison process.
    - It uses the `display` function from IPython to show the DataFrames.
    
    Examples
    --------
    >>> model.results(results_table=True, MAFE_val_df=True, head=5)
        Displays the results table and the MAFE validation DataFrame with the top 5 rows.
    """
    head = len(self.MAFE_val_df) if (head == True) else head
    dfs_list = [(results_table, self.results_table), (MAFE_val_df, self.MAFE_val_df), (MAFE_val_df_improve, self.MAFE_val_df_improve), (MSFE_val_df, self.MSFE_val_df), (MSFE_val_df_improve, self.MSFE_val_df_improve), (forecast_criteria_df, self.forecast_criteria_df), (lags_df, self.lags_df), (MAFE_train_df, self.MAFE_train_df), (MSFE_train_df, self.MSFE_train_df)]
    for dis, df in dfs_list:
        if dis:
            display(df.head(head))