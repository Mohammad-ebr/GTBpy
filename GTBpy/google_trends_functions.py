"""
Functions are used to get hsi from Google trends data
"""

#%%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from IPython.display import display
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize

from GTBpy.forecast_functions import *

#%%

def trend_loader(hpi, folder_path, RE_companies=None, drop_companies=True, title_offset=10):
    """
    Loads google trends data in csv format in a folder and returns four dataframes
    
    Parameters
    ----------
    hpi : Pandas dataframe of shape (n_samples, 1)
          HPI index data
            
    folder_path : str
                  a folder path in which google trends data are stored.
                  
    RE_companies : list of strs, default = None
                    Name of real estate companies selected to constitute aggregate search of companies index.
                    
    Returns
    -------
    df : pandas dataframe including HPI and all search queries
    df_full : pandas dataframe including HPI and search queries that have real value since 2004-01-01
    df_full_nonzero : pandas dataframe including HPI and search queries that have real value since 2004-01-01 and do not have values equal to zero
    df_part : pandas dataframe including HPI and search queries that are not in df_full
    """
    df = hpi.copy()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dft = pd.read_csv(file_path, skiprows=2, index_col=[0], parse_dates=[0], date_format='%Y-%m')
            if not isinstance(dft.index, pd.DatetimeIndex):
                dft = pd.read_csv(file_path, skiprows=2, index_col=[0], parse_dates=[0], date_format='%b-%y')
            if not isinstance(dft.index, pd.DatetimeIndex):
                dft = pd.read_csv(file_path, skiprows=2, index_col=[0], parse_dates=[0], date_format='%y-%b')
            dft.columns = [dft.columns[0][:-title_offset]]
            dft.index = dft.index.to_period('M').to_timestamp('M')
            dft.iloc[:,0] = dft.iloc[:,0].replace('<1', 0.5)
            # Exclude low search volume
            if sum(dft.iloc[:,0] == 0)/ len(dft) < 0.1:
                df = df.merge(dft, left_index=True, right_index=True, how='left')
    df = df.apply(pd.to_numeric, errors='coerce', axis=1).asfreq('ME')
    
    if RE_companies:
        df['RE_companies'] = df[RE_companies].sum(axis=1)
        if drop_companies:
            df = df.drop(columns=RE_companies)
    
    df_full = df.dropna(axis=1, subset=df.index[df.index > pd.Timestamp('2004-01-01')]).dropna()
    df_part = df.loc[:, ~df.columns.isin(df_full)].dropna()
    df_full_nonzero = df_full.loc[:, ~(df_full==0).any(axis=0)]
    return df, df_full, df_full_nonzero, df_part

#%%

def reg_trend_selector(series, train_cut, alpha, recursive):
    """
    Determine the appropriate trend model for a time series based on the Augmented Dickey-Fuller (ADF) test.

    Parameters
    ----------
    series : pd.Series
        The time series data to be analyzed.
    train_cut : str or pd.Timestamp
        The date or index that defines the end of the training period.
    alpha : float
        The significance level used to assess the p-value from the ADF test.
    recursive : bool
        If True, the function will recursively difference the series until stationarity is achieved, based on the ADF test.

    Returns
    -------
    series : pd.Series
        The original or differenced time series, depending on the ADF test results.
    trend_cols : list of str
        A list of columns corresponding to the appropriate trend model, which may include:
        - 'const' : Constant term
        - 't'     : Linear time trend
        - 't2'    : Quadratic time trend

    Notes
    -----
    - The function performs the ADF test on the training period of the series with different trend models ('c', 'ct', and 'ctt').
    - If the null hypothesis of the ADF test (that the series has a unit root) is not rejected at the given `alpha` level, the function recursively differences the series if `recursive` is set to True.
    - The trend model is chosen based on the significance of the ADF test with different trend components.

    Examples
    --------
    >>> series = pd.Series(np.random.randn(100).cumsum(), index=pd.date_range('2000-01-01', periods=100))
    >>> series, trend_cols = reg_trend_selector(series, train_cut='2000-12-31', alpha=0.05, recursive=True)
        The function returns the appropriately differenced series and the trend model to be used.
    """

    series = series.copy()
    
    series_train = series.loc[:train_cut]
    t, p, *_ = adfuller(series_train, regression='c') # tvalue(adf stat), pvalue, usedlag, nobs, critical values, icbest
    if p > alpha:
        t, p, *_ = adfuller(series_train, regression='ct')
        if p > alpha:
            t, p, *_ = adfuller(series_train, regression='ctt')
            if p > alpha:
                if recursive:
                    series.loc[:], trend_cols = reg_trend_selector(series.diff().dropna(), train_cut, alpha, recursive) # we should use .loc[:] in order to maintain the series length and have NaN for indexes that are not in the assign values in the right hand side (values coming from reg_trend_selector(series.diff().dropna(), train_cut, alpha, recursive))
                    order = sum(np.isnan(series))
                    print(f'Taking the {(order==1)*"first" + (order==2)*"second"} difference of {series.name}')
                else:
                    print(f"The ADF test for {series.name} was not rejected in any of the cases 'c', 'ct' and 'ctt'.")
            else:
                trend_cols = ['const', 't', 't2']
        else:
            trend_cols = ['const', 't']
    else:
        trend_cols = ['const']
                    
    return series, trend_cols

#%%

def reg_detrend_deseasonal(df, train_cut, deseasonal, detrend, regression='ct', alpha=0.1, recursive=True, col=None):
    """
    Detrend and/or deseasonalize a DataFrame using linear regression.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to be detrended and/or deseasonalized. Each column is treated as a separate time series.
    train_cut : str, int, float, or pd.Timestamp
        The date or index that defines the end of the training period. If a float between 0 and 1 is provided, it represents a fraction of the total dataset length.
    deseasonal : bool
        If True, the function removes the seasonal component from the series using monthly dummies.
    detrend : bool
        If True, the function removes the trend component from the series based on the specified regression model.
    regression : {'c', 'ct', 'ctt'}, optional
        The type of trend model to be used:
        - 'c' : Constant term only (default).
        - 'ct' : Constant and linear trend.
        - 'ctt' : Constant, linear, and quadratic trend.
        If None, the function automatically selects the trend model based on the Augmented Dickey-Fuller (ADF) test.
    alpha : float, optional
        The significance level used for the ADF test when automatically selecting the trend model. Default is 0.1.
    recursive : bool, optional
        If True, the function recursively differences the series until stationarity is achieved, based on the ADF test.
    col : list of str, optional
        A list of column names in `df` to be processed. If None, all columns are processed. Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the detrended and/or deseasonalized series. The trend and seasonal components are removed according to the specified parameters.

    Notes
    -----
    - The function adds a constant (`const`), linear time trend (`t`), and quadratic time trend (`t2`) to the DataFrame as potential regressors.
    - Monthly dummy variables are generated and used to remove seasonality if `deseasonal` is True.
    - The function determines the appropriate trend model using the ADF test if `regression` is set to None.
    - The function drops the additional columns (`const`, `t`, `t2`, and seasonal dummies) before returning the final DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'HPI': np.random.randn(100).cumsum()}, index=pd.date_range('2000-01-01', periods=100))
    >>> df_detrended = reg_detrend_deseasonal(df, train_cut='2005-01-01', deseasonal=True, detrend=True, regression='ct')
        This example removes both trend and seasonality from the 'HPI' series in `df` using a linear trend model with a constant and linear trend.
    """

    df = df.copy()
    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]

    cols = df.columns
    if col:
        cols = col
            
    time_index = np.arange(len(df))
    df = pd.concat([df, pd.DataFrame({'const':1, 't':time_index, 't2':time_index**2}, index=df.index)], axis=1)
    month = pd.Series(df.index.month, index=df.index)
    dummies = pd.get_dummies(month, dtype='float', drop_first=True)
    seasonal_cols = list(dummies.columns)
    df = pd.concat([df, dummies], axis=1)

    # This setup does not include "const" in deseasonalizing
    base_deduct_cols = []
    if deseasonal:
        base_deduct_cols = seasonal_cols
        deduct_cols = seasonal_cols
        
    if regression == 'c':
        trend_cols = ['const']
    elif regression == 'ct':    
        trend_cols = ['const', 't']
    elif regression == 'ctt':
        trend_cols = ['const', 't', 't2']
        
    for c in cols:
        if regression is None:
            df[c], trend_cols = reg_trend_selector(df[c], train_cut, alpha, recursive)

        if detrend:
            deduct_cols = base_deduct_cols + trend_cols

        reg_cols = trend_cols + seasonal_cols
        df_train = df.loc[:train_cut, [c] + reg_cols].dropna()
        params = sm.OLS(df_train[c], df_train[reg_cols]).fit().params
        df[c] -= (df[deduct_cols] @ params[deduct_cols]) # df[c] -= sm.OLS(df[c], df[deduct_cols]).predict(params[deduct_cols])
        # from statsmodels.regression.linear_model import RegressionResults
        # df[c] = RegressionResults(sm.OLS(df[c], df[deduct_cols]), params[deduct_cols]).resid
        
    df = df.drop(columns=['const', 't', 't2'] + seasonal_cols)
    
    return df

#%%

def MA_detrend_deseasonal(df, train_cut, deseasonal, detrend, col=None):
    """
    Detrend and/or deseasonalize a DataFrame using moving average decomposition.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to be detrended and/or deseasonalized. Each column is treated as a separate time series.
    train_cut : str, int, float, or pd.Timestamp
        The date or index that defines the end of the training period. If a float between 0 and 1 is provided, it represents a fraction of the total dataset length.
    deseasonal : bool
        If True, the function removes the seasonal component from the series using moving average decomposition.
    detrend : bool
        If True, the function removes the trend component from the series using moving average decomposition.
    col : list of str, optional
        A list of column names in `df` to be processed. If None, all columns are processed. Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the detrended and/or deseasonalized series. The trend and seasonal components are removed according to the specified parameters.

    Notes
    -----
    - The function uses `seasonal_decompose` from `statsmodels.tsa` to perform the moving average decomposition.
    - The `deseasonal` parameter removes the seasonal component using the seasonal decomposition from the training set.
    - The trend component is removed by subtracting the moving average trend calculated from the entire series.
    - The function creates a temporary column for month (`month`) to align seasonal adjustments, which is dropped before returning the final DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'HPI': np.random.randn(100).cumsum()}, index=pd.date_range('2000-01-01', periods=100))
    >>> df_ma_detrended = MA_detrend_deseasonal(df, train_cut='2005-01-01', deseasonal=True, detrend=True)
        This example removes both trend and seasonality from the 'HPI' series in `df` using moving average decomposition.
    """

    df = df.copy()
    
    cols = df.columns
    if col:
        cols = col
        
    df['month'] = df.index.month
    df_train = df.loc[:train_cut].copy()
    for c in cols:
        if detrend:
            df[c] -= sm.tsa.seasonal_decompose(df[c], two_sided=False).trend
        if deseasonal:
            seasonal = sm.tsa.seasonal_decompose(df_train[c], model='additive', two_sided=False).seasonal.iloc[:12]
            seasonal = pd.DataFrame(seasonal)
            seasonal['month'] = seasonal.index.month
            a = df[[c, 'month']].reset_index().merge(seasonal, how='left').set_index('Date')
            a.loc[:,c] -= a.loc[:,'seasonal']
            df[c] = a[c]
        
            
    return df.drop(columns=['month'])
    
#%%

def prepare_gtrends(df, detrend=False, deseasonal=False, method='reg', regression='ct', alpha=0.15, log=False, smooth=12, winsorize_trends=False, train_cut=0.8):
    """
    Prepares Google Trends data by applying optional transformations such as detrending, deseasonalizing, smoothing, and logging.
    There are two ways when deterend=False and deseasonal=True. 
    The first way is to just directly calculate the seasonal effect. 
    The other way is that first calculate the trend, subtract it from series and then calculate the seasonal factors. Then subtract the seasonal factor from the original series.
    In this way we include trend and residuals in the series that we return as a result, and exclude seasonal factors.
    In this function,'growth' method uses the first way and 'MA' and 'reg' methods use the second.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where the first column is the dependent variable (e.g., HPI) and the remaining columns are Google Trends data.
    detrend : bool, optional
        If True, the function removes the trend component from the Google Trends data.
    deseasonal : bool, optional
        If True, the function removes the seasonal component from the Google Trends data.
    method : str, optional
        The method used for detrending and/or deseasonalizing. Options are 'reg' for regression-based, 'MA' for moving average, or 'growth' for differencing. Default is 'reg'.
    regression : str, optional
        The type of regression to use for detrending. Options are 'c' (constant), 'ct' (constant + trend), or 'ctt' (constant + trend + quadratic trend). Used only if `method='reg'`. Default is 'ct'.
    alpha : float, optional
        The significance level for the augmented Dickey-Fuller test used in the regression-based method. Default is 0.15.
    log : bool, optional
        If True, log-transform the Google Trends data by applying `log(X + 1)` to avoid taking the logarithm of zero. Default is False.
    smooth : int or bool, optional
        The smoothing window size. If an integer is provided, it specifies the window size for the seasonal decomposition trend smoothing. If True, a default window size of 12 is used. Default is 12.
    winsorize_trends : bool, optional
        If True, winsorizes the Google Trends data by applying limits of 0.05 on both ends to reduce the impact of outliers. Default is False.
    train_cut : float or str or pd.Timestamp, optional
        The point at which to split the data into training and testing. If a float between 0 and 1 is provided, it represents a fraction of the total dataset length. If a string or timestamp is provided, it is used directly as a cutoff date. Default is 0.8.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the dependent variable and the transformed Google Trends data. The exact transformations depend on the provided parameters.

    Raises
    ------
    Exception
        If an invalid `method` is specified.

    Notes
    -----
    - The `method` parameter determines how the detrending and/or deseasonalizing is performed. 'reg' uses regression-based methods, 'MA' uses moving averages, and 'growth' uses differencing.
    - The `smooth` parameter can be used to apply additional smoothing using seasonal decomposition.
    - The `log` transformation is applied after winsorizing if both are enabled.
    - The final DataFrame returned is merged on the index of the dependent variable and the transformed Google Trends data, ensuring alignment.

    Examples
    --------
    >>> df = pd.DataFrame({'HPI': np.random.randn(100).cumsum(), 'trend': np.random.randn(100)}, index=pd.date_range('2000-01-01', periods=100))
    >>> prepared_df = prepare_gtrends(df, detrend=True, deseasonal=True, method='MA', smooth=6)
        This example removes trend and seasonality using the moving average method and applies a smoothing window of 6 periods to the 'trend' data.
    """
    df = df.copy()
    
    y = df.iloc[:,0]
    X = df.iloc[:,1:].copy()
    
    if winsorize_trends:
        X = X.apply(winsorize, limits=[0.05, 0.05], axis=0)
    
    if log:
        X = np.log(X + 1)
    
    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]
    
    if detrend | deseasonal:
        if method == 'MA':
            X = MA_detrend_deseasonal(X, train_cut=train_cut, deseasonal=deseasonal, detrend=detrend, col=None).dropna()
                
        elif method == 'reg':
            X = reg_detrend_deseasonal(X, train_cut=train_cut, deseasonal=deseasonal, detrend=detrend, regression=regression, alpha=alpha, recursive=True, col=None).dropna()
                
        elif method == 'growth':
            if detrend:
                X = X.diff().dropna()
            if deseasonal:
                X = reg_detrend_deseasonal(X, train_cut=train_cut, deseasonal=True, detrend=False, regression='c', alpha=alpha, recursive=False, col=None).dropna()
    
        else:
            raise Exception('"method" is not defined.')
    
    if smooth:
        smooth = 12 if (smooth==True) else smooth
        for c in X.columns:
            X[c] = sm.tsa.seasonal_decompose(X[c], two_sided=False, period=smooth).trend
        # X = X.rolling(window=smooth).mean()  # SMA
        # X = X.ewm(span=smooth).mean()  # EMA
        X = X.dropna()
    
    df = pd.merge(y, X, left_index=True, right_index=True, how='right')
    return df


#%%

def prepare_hpi(df, detrend=False, deseasonal=False, method='reg', regression='ct', alpha=0.15, log=False, train_cut=0.8):
    # ADF test stat is the tvalue of y_{t-1} in the regression, but the test pvalue is not the pvalue of coefficient of y_{t-1}.
    # This tvalue can change significantly with a small change of data since it uses large number of lags (12-15 lags in HPI case).
    # For example in regression='ct, for data range 2004-01 to 2019-12: t=-2.856 and p=0.177. For data range 2005-01 to 2019-12: t=-4.242 and p=0.0039 .
    # The start date of HPI can change based on whether we detrend gtrends by MA in the previous step.
    # That's why we can specify 'trend' option to deternd data without needing to reject ADF test's null.
    # You can access regression results by following methods:
        # t, p, cv, result = adfuller(df_train['HPI'], regression='ct', store=True, regresults=True)
        # result.autolag_results[lag].summary()
        # result.resols.summary()
    """
    Prepares the Housing Price Index (HPI) data by applying optional transformations such as detrending, deseasonalizing, and logging.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least a column labeled 'HPI', representing the Housing Price Index.
    detrend : bool, optional
        If True, removes the trend component from the HPI data. Default is False.
    deseasonal : bool, optional
        If True, removes the seasonal component from the HPI data. Default is False.
    method : str, optional
        The method used for detrending and/or deseasonalizing. Options are 'reg' for regression-based or 'MA' for moving average. Default is 'reg'.
    regression : str, optional
        The type of regression to use for detrending. Options are 'c' (constant), 'ct' (constant + trend), or 'ctt' (constant + trend + quadratic trend). Used only if `method='reg'`. Default is 'ct'.
    alpha : float, optional
        The significance level for the augmented Dickey-Fuller (ADF) test used in the regression-based method. Default is 0.15.
    log : bool, optional
        If True, log-transforms the HPI data by applying `log(HPI)`. Default is False.
    train_cut : float or str or pd.Timestamp, optional
        The point at which to split the data into training and testing. If a float between 0 and 1 is provided, it represents a fraction of the total dataset length. If a string or timestamp is provided, it is used directly as a cutoff date. Default is 0.8.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the transformed HPI data, depending on the selected options for detrending, deseasonalizing, and logging.

    Raises
    ------
    Exception
        If an attempt is made to detrend HPI using the Moving Average method.

    Notes
    -----
    - The function includes a log transformation option to stabilize variance, which is applied before any other transformations.
    - Detrending using the Moving Average (MA) method is explicitly disallowed for the HPI, as indicated by an exception.
    - The ADF test's significance level (`alpha`) and type of regression used can significantly impact the detrending process, particularly in borderline cases.

    Examples
    --------
    >>> df = pd.DataFrame({'HPI': np.random.randn(100).cumsum()}, index=pd.date_range('2000-01-01', periods=100))
    >>> prepared_df = prepare_hpi(df, detrend=True, deseasonal=True, method='reg', regression='ct')
        This example removes the trend and seasonality from the HPI data using a regression-based method with a constant and trend.
    """
    df = df.copy()
    
    if log:
        df['HPI'] = np.log(df['HPI'])
        
    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]

    if detrend | deseasonal:
        if method == 'reg':
            df['HPI'] = reg_detrend_deseasonal(df[['HPI']], train_cut=train_cut, deseasonal=deseasonal, detrend=detrend, regression=regression, alpha=alpha, recursive=False, col=None)['HPI']
        
        elif method == 'MA':
            if detrend:
                print('We are not allowed to detrend HPI by Moving Average method!!')
            
            if deseasonal:
                df['HPI'] = MA_detrend_deseasonal(df[['HPI']], train_cut=train_cut, deseasonal=deseasonal, detrend=False, col=None)['HPI']
                
    return df

#%%

def compute_weights(criteria, var_select, selection_index=None, coef_select='abs', n=10, power=4, rank_based=False):
    """
    Computes weights for a set of criteria, selecting top variables or indices based on the specified selection method.

    Parameters
    ----------
    criteria : pd.Series or pd.DataFrame
        The data containing the criteria used to compute weights, such as coefficients or cross-validation errors.
    var_select : str
        The selection method for the variables. Options include 'EN' (Elastic Net), 'coef' (coefficient), 'tvalue', 'CV' (cross-validation), or 'IC' (information criterion).
    selection_index : pd.Index or list, optional
        The indices to consider for selection. If None, the entire index of `criteria` is used. Default is None.
    coef_select : str, optional
        Method for selecting coefficients. Options are 'abs' for absolute values or 'neg' for negative values. Only relevant if `var_select` is 'EN', 'coef', or 'tvalue'. Default is 'abs'.
    n : int, optional
        The number of top variables or indices to select based on the computed weights. Default is 10.
    power : int or float, optional
        The exponent used to compute the weights. Higher powers give more weight to larger values (or smaller values if using `var_select='CV'` or `var_select='IC'`). Default is 4.
    rank_based : bool, optional
        If True, weights are computed based on the rank of each criterion rather than its raw value. Default is False.

    Returns
    -------
    weights : pd.Series
        A series of computed weights corresponding to the criteria.
    top_queries : pd.Index or list
        The top `n` indices or variables based on the computed weights.

    Raises
    ------
    ValueError
        If an invalid value for `var_select` or `coef_select` is provided.

    Notes
    -----
    - The function supports both positive and negative selection of coefficients, depending on the `coef_select` parameter.
    - When `rank_based` is True, the function ranks the criteria before computing the weights, which can be useful for stabilizing the selection process.
    - The function supports two main types of selection: those based on coefficient magnitudes ('EN', 'coef', 'tvalue') and those based on error metrics ('CV', 'IC').

    Examples
    --------
    >>> criteria = pd.Series([0.2, -0.5, 1.0, -0.3, 0.7], index=['A', 'B', 'C', 'D', 'E'])
    >>> weights, top_queries = compute_weights(criteria, var_select='coef', coef_select='abs', n=3)
    >>> print(weights)
    A    0.061926
    B    0.142322
    C    0.388067
    D    0.100637
    E    0.307048
    dtype: float64
    >>> print(top_queries)
    Index(['C', 'E', 'B'], dtype='object')

    >>> criteria = pd.Series([0.8, 0.5, 1.2, 0.9, 0.6], index=['A', 'B', 'C', 'D', 'E'])
    >>> weights, top_queries = compute_weights(criteria, var_select='CV', rank_based=True, n=2)
    >>> print(weights)
    A    0.214796
    B    0.355939
    C    0.131228
    D    0.178110
    E    0.119927
    dtype: float64
    >>> print(top_queries)
    Index(['B', 'D'], dtype='object')
    """
    if selection_index == None:
        selection_index = criteria.index
        
    if var_select in ['EN', 'coef', 'tvalue']:
        if coef_select == 'abs':
            criteria = np.abs(criteria)
        elif coef_select == 'neg':
            criteria = -criteria
        
        if rank_based:
            ranks = pd.Series(criteria).rank(ascending=False)
            weights = np.exp(-ranks)
            weights = weights / weights.sum()
        else:
            if np.min(criteria) < 0:
                criteria -= min(criteria) * 1.1 # criteria += abs(min(criteria)) + abs(max(criteria)) # criteria += abs(min(criteria)) + 1
            weights = (criteria ** power)/sum(criteria ** power)
        
        top_queries = (-criteria.loc[selection_index]).sort_values().index[:n]
        
    elif var_select in ['CV', 'IC']:
        if rank_based:
            ranks = pd.Series(criteria).rank(ascending=True)
            weights = np.exp(-ranks)
            weights = weights / weights.sum()
        else:
            if np.min(criteria) < 0:
                criteria -= min(criteria) * 1.1
            weights = (1/(criteria ** power)) / sum(1/(criteria ** power))
        
        top_queries = criteria.loc[selection_index].sort_values().index[:n]
        
    return weights, top_queries


#%%

def select_lag(criterias, var_select):
    if var_select in ['EN', 'coef', 'tvalue']:
        criteria = criterias.abs().groupby(lambda x: x.split('.L')[0]).max()
        index = criteria.abs().groupby(lambda x: x.split('.L')[0]).idxmax()
    elif var_select in ['CV', 'IC']:
        criteria = criterias.groupby(lambda x: x.split('.L')[0]).min()
        index = criterias.groupby(lambda x: x.split('.L')[0]).idxmin()

    return criteria, index

#%%

def scale_lag_X(X, train_cut, lagged, max_lag=None):

    X_train = X.loc[:train_cut]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit(X_train).transform(X), index=X.index, columns=X.columns)
    
    if lagged:
        X_scaled_lagged = pd.DataFrame()
        for feature in X_scaled.columns:
            for lag in range(max_lag):
                X_scaled_lagged[f'{feature}.L{lag}'] = X_scaled[feature].shift(lag)
        X_scaled = X_scaled_lagged.dropna()
        
    X_train_scaled = X_scaled.loc[:train_cut]
    
    return X_scaled, X_train_scaled

#%%

def reduce_criteria_expanded(criteria_dfs_expanded, var_select, lags, coef_select):
    
    only_criteria_df_full = criteria_dfs_expanded[[col for col in criteria_dfs_expanded.columns if col.startswith(var_select)]]
    
    if var_select in ['EN', 'coef', 'tvalue']:
        only_criteria_df = only_criteria_df_full.abs().T.groupby(lambda x: '_'.join(x.split('_')[:2])).max().T
        min_col_df = only_criteria_df_full.abs().T.groupby(lambda x: '_'.join(x.split('_')[:2])).idxmax().T
    elif var_select in ['CV', 'IC']:
        only_criteria_df = only_criteria_df_full.T.groupby(lambda x: '_'.join(x.split('_')[:2])).min().T
        min_col_df = only_criteria_df_full.T.groupby(lambda x: '_'.join(x.split('_')[:2])).idxmin().T
    min_col_df = min_col_df.replace(var_select, 'index', regex=True)
    
    index_df = pd.DataFrame(index=only_criteria_df.index, columns=[col.replace(var_select, 'index') for col in only_criteria_df.columns])
    for i in criteria_dfs_expanded.index:
        index_df.loc[i] = criteria_dfs_expanded.loc[i, min_col_df.loc[i]].values
    criteria_dfs = pd.concat([only_criteria_df, index_df], axis=1)
    # criteria_dfs = criteria_dfs[[col for pair in zip(only_criteria_df.columns, index_df.columns) for col in pair]]
    
    for lag in lags:
        weights, _ = compute_weights(criteria=criteria_dfs.loc[:, f'{var_select}_{lag}'], var_select=var_select, coef_select=coef_select, power=3, rank_based=False)
        criteria_dfs.loc[:, f'weight_{lag}'] = weights
        
    criteria_dfs = criteria_dfs[[f'{title}_{lag}' for lag in lags for title in [var_select, 'index', 'weight']]]
    
    return criteria_dfs

#%%

def pca_hsi_dif_lags(df, hsi_dfs, X_scaled_lagged_full, criteria_dfs, old_best, input_pool=True, auto_layer=False, layer=1, max_lag=3, var_select='EN', random=True, seed=None, cv=5, shuffle=False, n_iter=20, n_hsi=10, train_cut=0.8, alphas=np.logspace(-3,0,100), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], lags=[0,1,3,12], n=10, coef_select='abs', criteria_dis=True, verbose=False):

    df = df.copy()
    rng = np.random.default_rng(seed)
    
    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]
        
    y = df.iloc[max_lag-1:, 0]
    y_train = y.loc[:train_cut]

    X = df.iloc[:, 1:]
    X_scaled_lagged, X_train_scaled_lagged = scale_lag_X(X, train_cut=train_cut, lagged=True, max_lag=max_lag)
    X_scaled_lagged_full = pd.concat([X_scaled_lagged_full, X_scaled_lagged], axis=1).dropna()
    X_train_scaled_lagged_full = X_scaled_lagged_full.loc[:train_cut]
    
    if (var_select == 'EN') & input_pool:
        X_scaled_lagged, X_train_scaled_lagged = X_scaled_lagged_full, X_train_scaled_lagged_full
    
    if input_pool:
        selection_index = hsi_dfs.columns[1:]
    else:
        selection_index = df.columns[1:]

    hsi = pd.DataFrame(index=X.index[max_lag-1:])
    criteria_dfs = pd.concat([criteria_dfs, pd.DataFrame(index=df.columns[1:])])
    criteria_df_lagged = pd.DataFrame(index=X_scaled_lagged.columns)
    trends_count_df = pd.DataFrame()
    
    for lag in lags:
        trends_count = []
        y_adj = y_train.iloc[lag:]
        X_adj = X_train_scaled_lagged.shift(lag).dropna()
        
        criteria_lagged = np.array([])
        
        if var_select == 'EN':
            elastic_net_cv = ElasticNetCV(fit_intercept=True, alphas=alphas, l1_ratio=l1_ratio, cv=4, max_iter=20000)
            elastic_net_cv.fit(X_adj, y_adj) # I don't know if it is better to include all exogs in the ElasticNet or just the exog of the last layer
            criteria_lagged = elastic_net_cv.coef_

        elif var_select in ['coef', 'tvalue']:
            for feature in X_adj.columns:
                model = sm.OLS(y_adj, sm.add_constant(X_adj[[feature]])).fit()
                if var_select == 'coef':
                    criteria_lagged = np.append(criteria_lagged, model.params.iloc[-1])
                elif var_select == 'tvalue':
                    criteria_lagged = np.append(criteria_lagged, model.tvalues.iloc[-1])
        
        elif var_select in ['CV', 'IC']:
            for feature in X_adj.columns:
                ic = compute_ic_cv(y_adj, X_adj[[feature]], metric=var_select, fit_intercept=True, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
                criteria_lagged = np.append(criteria_lagged, ic)
            
        criteria_df_lagged.loc[X_adj.columns, f'{var_select}_{lag}'] = criteria_lagged
        criteria, index = select_lag(criteria_df_lagged[f'{var_select}_{lag}'], var_select=var_select)
        criteria_dfs.loc[X.columns, f'{var_select}_{lag}'] = criteria
        criteria_dfs.loc[X.columns, f'index_{lag}'] = index
        
        weights_full, _ = compute_weights(criteria=criteria_dfs[f'{var_select}_{lag}'], var_select=var_select, coef_select=coef_select, power=4, rank_based=False)
        criteria_dfs[f'weight_{lag}'] = weights_full
        
        top_queries = criteria_dfs.loc[selection_index].sort_values(f'weight_{lag}', ascending=False)[f'index_{lag}'].iloc[:n]

        if random:
            for j in range(1, n_hsi+1):
                selection_weights = criteria_dfs.loc[selection_index, f'weight_{lag}'] / criteria_dfs.loc[selection_index, f'weight_{lag}'].sum()
                top_queries = rng.choice(criteria_dfs.loc[selection_index, f'index_{lag}'], size=n, replace=False, p=selection_weights)
                    
                # Select the n queries with the highest coefficients by their absolute(?) values   
                X_scaled_selected = X_scaled_lagged_full.loc[:, top_queries]
                X_train_scaled_selected = X_train_scaled_lagged_full.loc[:, top_queries]
                trends_count += list(X_scaled_selected.columns)
                
                if verbose:
                    if var_select == 'EN':
                        print(f'hsi_{layer}_{lag}_{j},  alpha:{elastic_net_cv.alpha_},  l1_ratio: {elastic_net_cv.l1_ratio_}')
                    print(f'{n} Selected queries for hsi_{layer}_{lag}_{j}:\n{np.array(X_scaled_selected.columns)}')

                # Apply PCA
                pca = PCA(n_components=1)
                principal_component = pca.fit(X_train_scaled_selected).transform(X_scaled_selected)
                hsi[f'hsi_{layer}_{lag}_{j}'] = principal_component
            
            values, counts = np.unique(trends_count, return_counts=True)
            a = pd.DataFrame({f'count_{lag}':counts}, index=values)
            trends_count_df = pd.concat([trends_count_df, a], axis=1, sort=False).fillna(0).astype(int)
            trends_count_df = trends_count_df.sort_values(trends_count_df.columns[-1], ascending=False)
                
        else:
            # Select the n queries with the highest coefficients by their absolute(?) values
            X_scaled_selected = X_scaled_lagged_full.loc[:, top_queries]
            X_train_scaled_selected = X_train_scaled_lagged_full.loc[:, top_queries]
            
            if verbose:
                if var_select == 'EN':
                    print(f'hsi_{lag},  alpha:{elastic_net_cv.alpha_},  l1_ratio:{elastic_net_cv.l1_ratio_}')
                print(f'{n} Selected queries for hsi_{lag}:\n{np.array(X_scaled_selected.columns)}')

            # Apply PCA
            pca = PCA(n_components=1)
            principal_component = pca.fit(X_train_scaled_selected).transform(X_scaled_selected)
            hsi[f'hsi_{layer}_{lag}'] = principal_component
            
        
    if (verbose > 1) & bool(random):
        display(trends_count_df)
    
    criteria_dfs = criteria_dfs.sort_values(f'weight_{max(lags)}', ascending=False, key=abs)
    
    if criteria_dis:
        display(criteria_dfs)
    
    # `principal_component` now contains the first principal component of the selected search queries
    df = pd.merge(y, hsi, left_index=True, right_index=True, how='right')
    if var_select == 'EN':
        hsi_dfs = df
    else:
        hsi_dfs = pd.concat([hsi_dfs, hsi], axis=1, join='inner')
    
    if auto_layer:
        new_best = criteria_dfs[f'{var_select}_{max(lags)}'].iloc[0]
        layer += 1
        if new_best != old_best:
            df, hsi_dfs, X_scaled_lagged_full, criteria_dfs = pca_hsi_dif_lags(df, hsi_dfs, X_scaled_lagged_full, criteria_dfs, new_best, input_pool=input_pool, auto_layer=auto_layer, layer=layer, max_lag=max_lag, var_select=var_select, random=random, seed=seed, cv=cv, shuffle=shuffle, n_iter=n_iter, n_hsi=n_hsi, train_cut=train_cut, alphas=alphas, l1_ratio=l1_ratio, lags=lags, n=n, coef_select=coef_select, criteria_dis=criteria_dis, verbose=verbose) 
    
    return df, hsi_dfs, X_scaled_lagged_full, criteria_dfs

#%%

def pca_hsi(df, hsi_dfs, criteria_dfs, criteria_dfs_expanded, old_best, input_pool=True, auto_layer=False, layer=1, var_select='EN', random='intc', seed=None, cv=5, shuffle=False, n_iter=20, n_hsi=10, train_cut=0.8, alphas=np.logspace(-3,0,100), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], lags=[0,1,3,12], n=10, coef_select='abs', criteria_dis=True, verbose=False):

    df = df.copy()
    rng = np.random.default_rng(seed)
    
    if isinstance(train_cut, (int, float)):
        train_cut = df.index[round(len(df) * train_cut)-1]
        
    if (var_select in ['IC', 'coef', 'tvalue']) & (random=='intc'):
        random = 'rnd'
        print('"random" argument turned from "intc" to "rnd"')
        
    y = df.iloc[:,0]
    y_train = y.loc[:train_cut]
    
    _, X_train_scaled = scale_lag_X(df.iloc[:,1:], train_cut=train_cut, lagged=False)
    X_full_scaled, X_full_train_scaled = scale_lag_X(hsi_dfs.iloc[:,1:], train_cut=train_cut, lagged=False)
    
    if (var_select == 'EN') & input_pool:
        X_train_scaled = X_full_train_scaled
    
    if input_pool:
        selection_index = hsi_dfs.columns[1:]
    else:
        selection_index = df.columns[1:]
        
    hsi = pd.DataFrame(index=df.index)
    criteria_dfs = criteria_dfs.reindex(hsi_dfs.columns[1:])
    criteria_dfs_expanded = criteria_dfs_expanded.reindex(hsi_dfs.columns[1:])
    trends_count_df = pd.DataFrame()
    
    for lag in lags:
        trends_count = []
        X_adj = X_train_scaled.shift(lag).dropna()
        y_adj = y_train.iloc[lag:]
        
        
        criteria = np.array([])
        
        if not random == 'intc':
            if var_select == 'EN':
                elastic_net_cv = ElasticNetCV(fit_intercept=True, alphas=alphas, l1_ratio=l1_ratio, cv=4, max_iter=20000)
                elastic_net_cv.fit(X_adj, y_adj) # I don't know if it is better to include all exogs in the ElasticNet or just the exog of the last layer
                criteria = elastic_net_cv.coef_

            elif var_select in ['coef', 'tvalue']:
                for feature in X_adj.columns:
                    model = sm.OLS(y_adj, sm.add_constant(X_adj[[feature]])).fit()
                    if var_select == 'coef':
                        criteria = np.append(criteria, model.params.iloc[-1])
                    elif var_select == 'tvalue':
                        criteria = np.append(criteria, model.tvalues.iloc[-1])
            
            elif var_select in ['CV', 'IC']:
                for feature in X_adj.columns:
                    ic = compute_ic_cv(y_adj, X_adj[[feature]], metric=var_select, fit_intercept=True, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
                    criteria = np.append(criteria, ic)
                
            criteria_dfs.loc[X_adj.columns, f'{var_select}_{lag}'] = criteria
            weights_full, top_queries = compute_weights(criteria=criteria_dfs[f'{var_select}_{lag}'], selection_index=selection_index, var_select=var_select, coef_select=coef_select, power=3, rank_based=False)
            criteria_dfs[f'weight_{lag}'] = weights_full         

        if random:
            for j in range(1, n_hsi+1):
                criteria = np.array([])
                if random == 'intc':
                    if var_select == 'EN':
                        df_adj = pd.concat([y_adj.reset_index(drop=True), X_adj.reset_index(drop=True)], axis=1)
                        shuffle_df_adj = df_adj.sample(frac=1)
                        y_adj = shuffle_df_adj.iloc[:,0]
                        X_adj = shuffle_df_adj.iloc[:,1:]
                        elastic_net_cv = ElasticNetCV(fit_intercept=False, alphas=alphas, l1_ratio=l1_ratio, cv=4, max_iter=1000)
                        elastic_net_cv.fit(X_adj, y_adj)
                        criteria = elastic_net_cv.coef_

                    elif var_select == 'CV':
                        for feature in X_adj.columns:
                            ic = compute_ic_cv(y_adj, X_adj[[feature]], metric=var_select, fit_intercept=False, cv=cv, shuffle=shuffle, n_iter=n_iter, seed=seed)
                            criteria = np.append(criteria, ic)
                    
                    criteria_dfs_expanded.loc[X_adj.columns, f'{var_select}_{lag}_{j}'] = criteria
                    _, top_queries = compute_weights(criteria=criteria_dfs_expanded[f'{var_select}_{lag}_{j}'], selection_index=selection_index, var_select=var_select, coef_select=coef_select, power=3, rank_based=False)
                
                elif random == 'rnd':
                    selection_weights = criteria_dfs.loc[selection_index, f'weight_{lag}'] / criteria_dfs.loc[selection_index, f'weight_{lag}'].sum()
                    top_queries = rng.choice(selection_index, size=n, replace=False, p=selection_weights)
                    
                # Select the n queries with the highest coefficients by their absolute(?) values    
                X_scaled_selected = X_full_scaled[top_queries]
                X_train_scaled_selected = X_full_train_scaled[top_queries]
                trends_count += list(X_scaled_selected.columns)
                
                if verbose:
                    if var_select == 'EN':
                        print(f'hsi_{layer}_{lag}_{j},  alpha:{elastic_net_cv.alpha_},  l1_ratio: {elastic_net_cv.l1_ratio_}')
                    print(f'{n} Selected queries for hsi_{layer}_{lag}_{j}:\n{np.array(X_scaled_selected.columns)}')

                # Apply PCA
                pca = PCA(n_components=1)
                principal_component = pca.fit(X_train_scaled_selected).transform(X_scaled_selected)
                hsi[f'hsi_{layer}_{lag}_{j}'] = principal_component
            
            values, counts = np.unique(trends_count, return_counts=True)
            a = pd.DataFrame({f'count_{lag}':counts}, index=values)
            trends_count_df = pd.concat([trends_count_df, a], axis=1, sort=False).fillna(0).astype(int)
            trends_count_df = trends_count_df.sort_values(trends_count_df.columns[-1], ascending=False)
                
        else:
            # Select the n queries with the highest coefficients by their absolute(?) values
            X_scaled_selected = X_full_scaled[top_queries]
            X_train_scaled_selected = X_full_train_scaled[top_queries]
            
            if verbose:
                if var_select == 'EN':
                    print(f'hsi_{lag},  alpha:{elastic_net_cv.alpha_},  l1_ratio:{elastic_net_cv.l1_ratio_}')
                print(f'{n} Selected queries for hsi_{lag}:\n{np.array(X_scaled_selected.columns)}')

            # Apply PCA
            pca = PCA(n_components=1)
            principal_component = pca.fit(X_train_scaled_selected).transform(X_scaled_selected)
            hsi[f'hsi_{layer}_{lag}'] = principal_component
            
    if (verbose > 1) & bool(random):
        display(trends_count_df)
        if random == 'intc':
            display(criteria_dfs_expanded)
    
    if random == 'intc':
        criteria_dfs = criteria_dfs_expanded.T.groupby(lambda x: '_'.join(x.split('_')[:2])).mean().T
        for lag in lags:
            weights, _ = compute_weights(criteria=criteria_dfs.loc[:, f'{var_select}_{lag}'], var_select=var_select, coef_select=coef_select, power=3, rank_based=False)
            criteria_dfs[f'weight_{lag}'] = weights
        criteria_dfs = criteria_dfs[[f'{title}_{lag}' for lag in lags for title in [var_select, 'weight']]]
        
        criteria_dfs_expanded = criteria_dfs_expanded.sort_values(criteria_dfs_expanded.columns[-1])
    
    criteria_dfs = criteria_dfs.sort_values(criteria_dfs.columns[-1], ascending=False, key=abs)
    
    if criteria_dis:
        display(criteria_dfs)
        
    
    # `principal_component` now contains the first principal component of the selected search queries
    df = pd.merge(y, hsi, left_index=True, right_index=True, how='right')    
    hsi_dfs = pd.concat([hsi_dfs, hsi], axis=1, sort=False)
    
    if auto_layer:
        new_best = criteria_dfs[f'{var_select}_{max(lags)}'].iloc[0]
        layer += 1
        if new_best != old_best:
            hsi_dfs, criteria_dfs = pca_hsi(df, hsi_dfs, criteria_dfs, criteria_dfs_expanded, new_best, input_pool=input_pool, auto_layer=auto_layer, layer=layer, var_select=var_select, random=random, seed=seed, cv=cv, shuffle=shuffle, n_iter=n_iter, n_hsi=n_hsi, train_cut=train_cut, alphas=alphas, l1_ratio=l1_ratio, lags=lags, n=n, coef_select=coef_select, criteria_dis=criteria_dis, verbose=verbose) 
    
    return hsi_dfs, criteria_dfs

#%%

def GT_plot(self, df, exog, lag, title=None, winsorize_trend=False, scaled=False):
    """
    Generate plots to visualize the relationship between the Housing Price Index (HPI) and a selected exogenous variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the HPI and exogenous variables.
    exog : str
        The name of the exogenous variable to be plotted and analyzed.
    lag : int
        The lag to apply to the exogenous variable in the scatter plot and time series plot.
    title : str, optional
        The title for the overall figure (default is None).
    winsorize_trend : bool, optional
        If True, apply winsorization to the exogenous variable to limit the effect of outliers (default is False).
    scaled : bool, optional
        If True, standardize the exogenous variable by removing the mean and scaling to unit variance (default is False).

    Returns
    -------
    None
        Displays a 2x2 grid of plots:
        - Top-left: Time series plot of HPI.
        - Top-right: Scatter plot of lagged exogenous variable vs. HPI.
        - Bottom-left: Time series plot of the lagged exogenous variable.
        - Bottom-right: (Removed) was intended for an additional plot, now removed.

    Notes
    -----
    - The function visualizes the relationship between HPI and a chosen exogenous variable by generating time series and scatter plots.
    - If `winsorize_trend` is True, the exogenous variable is adjusted to mitigate the impact of extreme values.
    - If `scaled` is True, the exogenous variable is standardized before being plotted.
    - The bottom-right subplot (`ax4`) is intentionally removed, leaving three plots in the 2x2 grid.
    
    Examples
    --------
    >>> model.GT_plot(df, exog='Google_Trends_Search', lag=2, title='Google Trends vs. HPI', winsorize_trend=True, scaled=True)
        Displays the plots for the relationship between HPI and the Google Trends search data with a lag of 2.
    """
    plt.rcParams.update({'mathtext.default': 'regular'})
    # Instead of the above line, we can use terms $\mathregular{N_i}$ and $\mathrm{N_i}$ in setting the label of x-axis and y-axis.
    # You can one example for ax3 in GT_plot() function
    
    df = df.copy()
    y = df[['HPI']]
    X = df[[exog]]
    if winsorize_trend:
        X = X.apply(winsorize, limits=[0.05,0.05], axis=0)
    if scaled:
        # Standardize the features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit(X.loc[:self.train_cut]).transform(X), index=X.index, columns=X.columns)
    
    y_adj = y.iloc[lag:]
    X_adj = X.shift(lag).dropna()
    y_adj_train = y_adj.loc[:self.train_cut]
    y_adj_val = y_adj.loc[self.train_cut:].iloc[1:]
    X_adj_train = X_adj.loc[:self.train_cut]
    X_adj_val = X_adj.loc[self.train_cut:].iloc[1:]
    
    fig, ((ax1, ax2),( ax3, ax4)) = plt.subplots(2,2, figsize=(16,8)) # figsize=(16,9) | figsize=(16,8)
    if title:
        fig.suptitle(title)
    
    ax1.plot(y)
    ax3.plot(X_adj)
    ax2.scatter(X_adj_train, y_adj_train, color='#1f77b4', label='Train Sample')
    ax2.scatter(X_adj_val, y_adj_val, color='#ff7f0e', label='Validation Sample')
    
    ax1.set_title('HPI')
    ax1.set_ylabel('$HPI_{t}$')
    ax1.axvline(x = pd.Timestamp(self.train_cut), color = 'r', ls='--', label = 'train cut')
    ax1.legend()
    
    ax3.set_title(exog)
    escaped_exog = exog.replace('_', '\_').replace(' ', '\ ')
    laged_exog = f'${escaped_exog}_{{t-{lag}}}$'
    ax3.set_ylabel(laged_exog)
    # ax3.set_ylabel(f'$\mathregular{{{escaped_exog}_{{t-{lag}}}}}$')
    ax3.set_xlabel('Date')
    ax3.axvline(x = pd.Timestamp(self.train_cut), color = 'r', ls='--', label = 'train cut')
    ax3.legend()
    ax3.sharex(ax1)
    
    ax2.set_ylabel('$HPI_{t}$')
    ax2.set_title('Scatter Plot')
    ax2.set_xlabel(laged_exog)
    ax2.legend()
    
    fig.delaxes(ax4)
    
#%%

def plot_gtrends(self, exog, h:int, winsorize_trend=False, scaled=False):
    """
    Plot Google Trends data and its transformations over different stages of processing.

    Parameters
    ----------
    exog : str or None
        The exogenous variable (Google Trends search term) to be plotted. If None, the variable with the highest weight for the specified horizon `h` is selected.
    h : int
        The forecast horizon to be used. If `h` is not in `self.selection_h_list`, the maximum value from `self.selection_h_list` is used.
    winsorize_trend : bool, optional
        If True, the Google Trends data is winsorized (default is False).
    scaled : bool, optional
        If True, the Google Trends data is scaled using standardization (default is False).

    Returns
    -------
    None
        Displays a series of plots showing the original Google Trends data, winsorized data, prepared Google Trends data, and the corresponding Housing Price Index (HPI) data.

    Notes
    -----
    - This function visualizes the Google Trends data (`exog`) across different stages: original, after winsorization, after preparation, and after scaling.
    - The `h` parameter specifies the forecast horizon, and the function adjusts the plots accordingly.
    - If `exog` is not provided, the function automatically selects the best-performing Google Trends term based on predefined criteria.
    - The vertical red dashed line in each plot indicates the cut-off point between training and validation data.

    Examples
    --------
    >>> plot_gtrends(exog='housing_market', h=6, winsorize_trend=True, scaled=True)
        Plots the Google Trends data for 'housing_market' with winsorization and scaling applied at horizon 6.

    >>> plot_gtrends(exog=None, h=3)
        Automatically selects the best Google Trends term and plots it at horizon 3 without any additional processing.
    """

    h = h if h in self.selection_h_list else max(self.selection_h_list)
    if not self.winsorize_trend:
        winsorize_trend = False
        
    if exog is None:
        gtrends_criteria = self.criteria_dfs.loc[self.gtrends_df.columns[1:],:]
        exog = gtrends_criteria[f'weight_{h}'].idxmax()
    
    GT_plot(self, df=self.df, exog=exog, lag=h, title='Original data')
    if winsorize_trend:
        GT_plot(self, df=self.df, exog=exog, lag=h, title='After winsorize gtrend', winsorize_trend=True)
    GT_plot(self, df=self.gtrends_df, exog=exog, lag=h, title='After gtrend preparation')
    GT_plot(self, df=self.hpi_df, exog=exog, lag=h, title='After HPI preparation')
    if scaled:
        GT_plot(self, df=self.hpi_df, exog=exog, lag=h, title='Scaled gtrends', scaled=True)
        
#%%

def plot_hsi(self, hsi=None):
    """
    Plot the selected Housing Search Index (HSI) for different forecast horizons.

    Parameters
    ----------
    hsi : str, optional
        The specific HSI to be plotted. If None, the HSI with the highest weight for each forecast horizon (`h`) is selected (default is None).

    Returns
    -------
    None
        Displays the plots for the selected HSIs across the specified forecast horizons in `self.selection_h_list`.

    Notes
    -----
    - This method generates a plot for each forecast horizon (`h`) in reverse order (starting from the longest horizon).
    - If no specific `hsi` is provided, the method will automatically select and plot the HSI with the maximum weight for each `h` from `self.criteria_dfs`.
    - The plots are created using the `GT_plot` function, which visualizes the relationship between the HPI and the selected HSI.
    
    Examples
    --------
    >>> model.plot_hsi(hsi='Sentiment_Index')
        Plots the specified 'Sentiment_Index' for all horizons in the selection list.
        
    >>> model.plot_hsi()
        Automatically selects and plots the HSI with the maximum weight for each horizon in the selection list.
    """
    n = len(self.selection_h_list)
    for i in range(n-1, -1, -1):
        h = self.selection_h_list[i]
        hsi = self.criteria_dfs[f'weight_{h}'].idxmax() if hsi is None else hsi
        GT_plot(self, self.hsi_dfs, title=f'hsi={hsi}, h={h}', exog=hsi, lag=h)

#%%

def plot_improvement(obj_list, measure='MAFE', h_list=None, colors=None, labels=None, figsize=(15,9)):
    """
    Plot the Kernel Density Estimate (KDE) of improvement measures (MAFE/MSFE) for different models across forecast horizons.

    Parameters
    ----------
    obj_list : list
        A list of objects containing the results of the models to be compared. Each object should have `MAFE_val_df_improve` and `MSFE_val_df_improve` attributes.
    measure : str, optional
        The performance measure to be plotted. Can be 'MAFE' (Mean Absolute Forecast Error) or 'MSFE' (Mean Squared Forecast Error). Default is 'MAFE'.
    h_list : list, optional
        The list of forecast horizons to be considered. If None, the horizons from the first object in `obj_list` will be used.
    colors : list, optional
        List of colors to be used for plotting different models. If None, a default list of colors will be used.
    labels : list, optional
        List of labels for the models in `obj_list`. If None, default labels ('G1', 'G2', ...) will be generated.
    figsize : tuple, optional
        Size of the figure. Default is (15, 9).

    Returns
    -------
    None
        Displays the KDE plots for the selected measure across the specified forecast horizons.

    Notes
    -----
    - This function plots the density of the percentage improvement in MAFE/MSFE for different models over multiple forecast horizons (`h`).
    - A vertical line is added to each plot to mark the performance of the Universal Housing Sentiment Index (UHSI) for the corresponding forecast horizon.
    - If `h_list` is not provided, the forecast horizons from the first model in `obj_list` are used by default.
    
    Examples
    --------
    >>> plot_improvement(models_list, measure='MAFE', h_list=[1, 3, 6, 12], colors=['blue', 'orange'], labels=['Model 1', 'Model 2'])
        Plots the MAFE improvement KDE for the specified models and forecast horizons.
    """

    plt.rcParams.update({'mathtext.default': 'regular'})
    
    if not h_list:
        h_list = obj_list[0].h_list
    if not colors:
        colors = ['blue', 'orange', 'red', 'green', 'purple', 'yellow']
    if not labels:
        labels = [f'G{i}' for i in range(1, len(obj_list)+1)]
    
    if len(h_list) == 1:
        fig, axs = plt.subplots(1,1, sharex=False, sharey=False, figsize=figsize)
    else:
        fig, axs = plt.subplots(math.ceil(len(h_list)/2),2, sharex=False, sharey=False, figsize=figsize)
        
    # fig.suptitle(f'{measure} Improvement KDE Plot')
    fig.supxlabel(f'{measure} Improvement in percent')
    fig.supylabel('Density')
    axs = axs.flatten() if len(h_list) > 1 else [axs]
    
    if measure == 'MAFE':
        df_list = [obj.MAFE_val_df_improve for obj in obj_list]
    elif measure == 'MSFE':
        df_list = [obj.MSFE_val_df_improve for obj in obj_list]
        
    for i, h in enumerate(h_list):
        for j, df in enumerate(df_list):
            sns.kdeplot(df.iloc[1:,i], ax=axs[i], fill=True, label=f'${labels[j]}\ B_{{{h}}}$', color=colors[j])
            axs[i].axvline(x = df.iloc[:,i].loc[f'UHSI_{h}'], color = colors[j], ls='-', label = f'{labels[j]} UHSI_{h}')
            
        axs[i].set_title(f'h={h}')
        axs[i].set_xlabel('') # axs[i].xaxis.label.set_visible(False)
        axs[i].set_ylabel('')
        axs[i].legend(loc='upper left', prop={'size': 9})
        
#%%

def plot_forecast(self, exog):
    """
    Plot the forecasted vs actual Housing Price Index (HPI) values over different forecast horizons.

    Parameters
    ----------
    exog : str or list, optional
        The exogenous variable(s) used in the forecasting models. If None, the best performing exogenous variable for each horizon is selected based on the criteria in `self.forecast_criteria_df`.

    Returns
    -------
    None
        Displays the forecast plots for the actual and predicted HPI across different forecast horizons.

    Notes
    -----
    - The function generates subplots for each forecast horizon `h` in `self.h_list`, displaying the actual and predicted HPI.
    - If `exog` is a list, the corresponding element is used as the exogenous variable for each horizon. If `exog` is a single string, it is used for all horizons.
    - The vertical red dashed line marks the cut-off point between the training and validation datasets.

    Examples
    --------
    >>> model.plot_forecast(exog=['UHSI_1', 'UHSI_3', 'UHSI_6'])
        Plots the forecasts using the specified exogenous variables for each horizon.

    >>> model.plot_forecast(exog='UHSI_3')
        Plots the forecasts using 'UHSI_3' as the exogenous variable for all horizons.
    """

    if exog is None:
        exog = list(self.forecast_criteria_df.apply(lambda x: x.idxmin()))
    
    fig, axs = plt.subplots(math.ceil(len(self.h_list)/2),2, sharex=True, sharey=True, figsize=(15,9))
    fig.suptitle('Forecast')
    fig.supxlabel('Date')
    fig.supylabel('log HPI')
    if self.original_scale:
        fig.supylabel('HPI')
    axs = axs.flatten()
    
    for j, h in enumerate(self.h_list):
        Y = self.forecast_dict[h].loc[:,'HPI']
        model_exog = exog[j] if isinstance(exog, list) else exog
        Y_pred = self.forecast_dict[h].loc[:,model_exog]
        axs[j].plot(Y, label='Actual')
        axs[j].plot(Y_pred, label='Prediction')
        axs[j].set_title('h=' + str(h) + ', exog=' + model_exog)
        axs[j].axvline(x = pd.Timestamp(self.train_cut), color = 'r', ls='--', label = 'train cut')
        axs[j].legend()






# ******************* Google Trend Classes ***************************
#%%

class GoogleTrend():
    """
    A class for processing and modeling Google Trends data to predict housing prices.
    This class provides tools for preparing, transforming, and modeling time series data using methods such as detrending,
    deseasonalizing, principal component analysis (PCA), and lagged variable selection.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the initial dataset, including the Housing Price Index (HPI) and Google Trends data.
    train_cut : float, int, or str, optional
        Defines the end of the training period. If a float between 0 and 1 is provided, it represents the fraction of the dataset used for training. Default is 0.8.
    verbose : bool, optional
        If True, enables detailed output for logging purposes. Default is False.
    seed : int or None, optional
        Seed for random number generation, used to ensure reproducibility. Default is None.
    pca_lag : bool, optional
        If True, PCA with lagged variables is used during the feature extraction. Default is True.

    Attributes
    ----------
    df : pd.DataFrame
        A copy of the initial dataset provided by the user.
    train_cut : pd.Timestamp
        Timestamp or index defining the end of the training period.
    verbose : bool
        Verbosity level for logging purposes.
    seed : int or None
        Seed for random number generation.
    pca_lag : bool
        Specifies whether PCA with lagged variables should be used.
    gtrends_df : pd.DataFrame
        Transformed Google Trends data.
    hpi_df : pd.DataFrame
        Transformed Housing Price Index (HPI) data.
    hsi_df : pd.DataFrame
        Final set of features generated after processing HPI and Google Trends data.
    hsi_dfs : pd.DataFrame
        Collection of transformed features including multiple PCA layers.
    criteria_dfs : pd.DataFrame
        DataFrame containing computed criteria values for feature selection.
    criteria_dfs_expanded : pd.DataFrame
        DataFrame containing expanded versions of computed criteria for feature selection.
    forecast_hsi : pd.DataFrame
        The final set of features used for forecasting.
    MAFE_val_df, MSFE_val_df, forecast_criteria_df, etc.
        DataFrames and metrics representing model performance during the forecast.

    Methods
    -------
    prepare_gtrends(detrend=True, deseasonal=True, ...)
        Prepares Google Trends data by applying optional transformations such as detrending and deseasonalizing.
    prepare_hpi(detrend=False, deseasonal=True, ...)
        Prepares the HPI data by applying optional transformations such as detrending and deseasonalizing.
    compute_hsi(layer=1, ...)
        Extracts key features from the dataset using PCA, lagged variables, and selection criteria.
    lag_setting(lag_select='IC', ...)
        Sets the lag selection criteria and related parameters for the model.
    forecast(h_list=[1,3,6,12], ...)
        Forecasts the HPI using prepared Google Trends data and selected lags.
    results(results_table=False, ...)
        Displays various results from the model, including performance metrics.
    plot_gtrends(exog=None, ...)
        Plots Google Trends data used in the model.
    plot_hsi(hsi=None)
        Plots HSI data used in the model.
    plot_improvement(measure='MAFE', ...)
        Plots improvements in forecasting accuracy based on specified performance measures.
    plot_forecast(exog=None)
        Plots the forecasted HPI based on the selected model features.
        
    Examples
    --------
    >>> GT_object = GoogleTrend(df, '2019-12-31', verbose=1, seed=1, pca_lag=True)
    >>> GT_object.prepare_gtrends(detrend=True, deseasonal=True, method='reg', regression='ct', log=False, smooth=12)
    >>> GT_object.prepare_hpi(detrend=True, deseasonal=True, method='reg', regression='ct', log=True)
    >>> GT_object.compute_hsi(layer=1, input_pool=True, auto_layer=False, n_input=10, n_hsi=20, max_lag=3, var_select='CV', random='rnd', cv=5, shuffle=True, n_iter=20, selection_h_list=[0,1,3,6,12])
    >>> GT_object.compute_hsi(layer=2, input_pool=True, auto_layer=True, n_input=10, n_hsi=20, max_lag=1, var_select='CV', random='rnd', cv=5, shuffle=True, n_iter=20, selection_h_list=[0,1,3,6,12])
    >>> GT_object.lag_setting(y_lags='Auto', exog_lags='Auto', max_lag=3, lag_select='IC')
    >>> GT_object.forecast(h_list=[1,3,6,12], seasonal=True, hsi_CV_select=True, fit_intercept=True, cv=5, n_iter=20, original_scale=False)
    >>> GT_object.results(MAFE_val_df=True, MAFE_val_df_improve=True, forecast_criteria_df=True, lags_df=True, head=True)
    """
    def __init__(self, df, train_cut=0.8, verbose=False, seed=None, pca_lag=True):
        self.df = df.copy()
        self.train_cut = train_cut
        self.verbose = verbose
        self.seed = seed
        self.pca_lag = pca_lag
        self.info = ''
        
        if isinstance(self.train_cut, (int, float)):
            self.train_cut = df.index[round(len(df) * self.train_cut)-1]
            
        self.X_scaled_lagged_full = pd.DataFrame()
        self.criteria_dfs = pd.DataFrame()
        self.criteria_dfs_expanded = pd.DataFrame()
    
    def prepare_gtrends(self, detrend=True, deseasonal=True, method='reg', regression='ct', alpha=0.15, log=False, smooth=False, winsorize=False):
        """
        Prepares Google Trends data by applying optional transformations such as detrending, deseasonalizing, smoothing, and logging.
        
        Parameters
        ----------
        detrend : bool, optional
            If True, removes the trend component from the Google Trends data. Default is True.
        deseasonal : bool, optional
            If True, removes the seasonal component from the Google Trends data. Default is True.
        method : str, optional
            The method used for detrending and/or deseasonalizing. Options are 'reg' for regression-based, 'MA' for moving average, or 'growth' for differencing. Default is 'reg'.
        regression : str, optional
            The type of regression to use for detrending. Options are 'c' (constant), 'ct' (constant + trend), or 'ctt' (constant + trend + quadratic trend). Used only if `method='reg'`. Default is 'ct'.
        alpha : float, optional
            The significance level for the augmented Dickey-Fuller (ADF) test used in the regression-based method. Default is 0.15.
        log : bool, optional
            If True, log-transforms the Google Trends data by applying `log(X + 1)` to stabilize variance. Default is False.
        smooth : bool or int, optional
            If True, applies smoothing using a default window of 12. If an integer is provided, it specifies the window size for smoothing. Default is False.
        winsorize : bool, optional
            If True, applies winsorization to the Google Trends data to reduce the effect of outliers. Default is False.

        Returns
        -------
        None
            Updates the `gtrends_df` attribute with the prepared Google Trends data after applying the specified transformations.

        Notes
        -----
        - This method prepares the Google Trends data for further modeling by optionally removing trend and seasonal components, smoothing, and log-transforming the data.
        - The `method` parameter determines the approach for detrending and deseasonalizing, allowing flexibility in how these transformations are applied.
        - The transformations help stabilize the data, improve stationarity, and reduce the impact of outliers, which can enhance the performance of downstream models.
        """
        
        self.winsorize_trend = winsorize
        self.gtrends_df = prepare_gtrends(self.df, detrend=detrend, deseasonal=deseasonal, method=method, regression=regression, alpha=alpha, log=log, smooth=smooth, winsorize_trends=winsorize, train_cut=self.train_cut)
    
    def prepare_hpi(self, detrend=False, deseasonal=True, method='reg', regression='ct', alpha=0.15, log=False):
        """
        Prepares the Housing Price Index (HPI) data by applying optional transformations such as detrending, deseasonalizing, and logging.

        Parameters
        ----------
        detrend : bool, optional
            If True, removes the trend component from the HPI data. Default is False.
        deseasonal : bool, optional
            If True, removes the seasonal component from the HPI data. Default is True.
        method : str, optional
            The method used for detrending and/or deseasonalizing. Options are 'reg' for regression-based or 'MA' for moving average. Default is 'reg'.
        regression : str, optional
            The type of regression to use for detrending. Options are 'c' (constant), 'ct' (constant + trend), or 'ctt' (constant + trend + quadratic trend). Used only if `method='reg'`. Default is 'ct'.
        alpha : float, optional
            The significance level for the augmented Dickey-Fuller (ADF) test used in the regression-based method. Default is 0.15.
        log : bool, optional
            If True, log-transforms the HPI data by applying `log(HPI)`. Default is False.

        Returns
        -------
        None
            Updates the `hpi_df` attribute with the prepared HPI data after applying the specified transformations.

        Notes
        -----
        - This method prepares the Housing Price Index (HPI) data for further modeling by optionally removing trend and seasonal components and applying a log transformation.
        - The `method` parameter determines how the detrending and deseasonalizing are applied, offering flexibility in transformations.
        - The transformations help in stabilizing the variance, improving stationarity, and making the data suitable for time series modeling and forecasting.
        """

        self.log = log
        self.original_hpi = None
        if detrend | deseasonal:
            self.original_hpi = self.df.HPI
        self.hpi_df = prepare_hpi(self.gtrends_df, detrend=detrend, deseasonal=deseasonal, method=method, regression=regression, alpha=alpha, log=log, train_cut=self.train_cut)
        self.hsi_df = self.hpi_df.copy()
        self.hsi_dfs = self.hpi_df.copy()
        
    def compute_hsi(self, layer=1, input_pool=True, auto_layer=False, n_input=10, n_hsi=20, max_lag=3, var_select='CV', random=False, cv=3, shuffle=True, n_iter=20, alphas=None, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], selection_h_list=[0,1,3,6,12], coef_select='abs', criteria_dis=False):
        """
        Extracts key features from the dataset using Principal Component Analysis (PCA), lagged variables, and feature selection criteria.

        Parameters
        ----------
        layer : int, optional
            The current layer of PCA and feature extraction. Default is 1.
        input_pool : bool, optional
            If True, the full input feature pool is used during the feature extraction. Default is True.
        auto_layer : bool, optional
            If True, additional layers are added automatically until no improvement is observed. Default is False.
        n_input : int, optional
            Number of input features to select. Default is 10.
        n_hsi : int, optional
            Number of HSI (Housing Sentiment Index) features to compute. Default is 20.
        max_lag : int, optional
            Maximum lag to consider for lagged features. Default is 3.
        var_select : str, optional
            The method used for variable selection. Options include 'CV', 'EN', 'coef', or 'tvalue'. Default is 'CV'.
        random : bool or str, optional
            If True or set to 'intc' or 'rnd', a randomized selection process is used. Default is False.
        cv : int, optional
            Number of cross-validation folds. Default is 3.
        shuffle : bool, optional
            If True, shuffle the data during cross-validation. Default is True.
        n_iter : int, optional
            Number of iterations for randomized search or selection. Default is 20.
        alphas : array-like, optional
            Array of alpha values for ElasticNetCV. Default is np.logspace(-3, 0, 100).
        l1_ratio : list, optional
            List of L1 ratios for ElasticNetCV. Default is [.1, .5, .7, .9, .95, .99, 1].
        selection_h_list : list of int, optional
            List of lag periods to consider for selection. Default is [0, 1, 3, 6, 12].
        coef_select : str, optional
            Method for selecting coefficients ('abs' for absolute values or 'neg' for negative values). Default is 'abs'.
        criteria_dis : bool, optional
            If True, displays criteria DataFrames used in feature selection. Default is False.

        Returns
        -------
        None
            Updates the attributes `hsi_df`, `hsi_dfs`, `X_scaled_lagged_full`, `criteria_dfs`, and `forecast_hsi` with the newly computed features and criteria.

        Notes
        -----
        - This method uses PCA for dimensionality reduction and lagged variable selection to generate new features.
        - Feature selection is based on different criteria, including ElasticNet coefficients, cross-validation metrics, and other statistics.
        - The process can include both deterministic and randomized approaches to ensure robustness.
        """
        if alphas is None:
            alphas = np.logspace(-3,0,100)
        self.var_select = var_select
        self.random = random
        self.selection_h_list = selection_h_list
        if self.pca_lag:
            self.hsi_df, self.hsi_dfs, self.X_scaled_lagged_full, self.criteria_dfs = pca_hsi_dif_lags(self.hsi_df, self.hsi_dfs, self.X_scaled_lagged_full, self.criteria_dfs, self.criteria_dfs_expanded, old_best=None, input_pool=input_pool, auto_layer=auto_layer, layer=layer, max_lag=max_lag, var_select=var_select, random=random, seed=self.seed, cv=cv, shuffle=shuffle, n_iter=n_iter, n_hsi=n_hsi, train_cut=self.train_cut, alphas=alphas, l1_ratio=l1_ratio, lags=selection_h_list, n=n_input, coef_select=coef_select, criteria_dis=criteria_dis, verbose=self.verbose)
        else:
            self.hsi_df, self.hsi_dfs, self.criteria_dfs = pca_hsi(self.hsi_df, self.hsi_dfs, self.criteria_dfs, self.criteria_dfs_expanded, old_best=None, input_pool=input_pool, auto_layer=auto_layer, layer=layer, var_select=var_select, random=random, seed=self.seed, cv=cv, shuffle=shuffle, n_iter=n_iter, n_hsi=n_hsi, train_cut=self.train_cut, alphas=alphas, l1_ratio=l1_ratio, lags=selection_h_list, n=n_input, coef_select=coef_select, criteria_dis=criteria_dis, verbose=self.verbose)
        
        if input_pool:
            self.forecast_hsi = self.hsi_dfs[['HPI'] + list(self.criteria_dfs.index[:(n_hsi*len(selection_h_list))])]
        else:
            if auto_layer:
                last_layer = max([int(col.split('_')[1]) for col in self.hsi_dfs.columns if col.startswith('hsi')])
                self.forecast_hsi = self.hsi_dfs[['HPI'] + [col for col in self.hsi_dfs.columns if col.startswith(f'hsi_{last_layer - 2}')]]
            else:
                self.forecast_hsi = self.hsi_df

    def lag_setting(self, y_lags='Auto', exog_lags='Auto', max_lag=3, lag_select='IC', fit_intercept=False, cv=5, shuffle=True, n_iter=20, var_order='cross'):
        """
        Sets the lag selection criteria and related parameters for the model.

        Parameters
        ----------
        lag_select : str, optional
            The method used for selecting the lag order. Options include 'IC' (information criterion), 'CV' (cross-validation), etc. Default is 'IC'.
        fit_intercept : bool, optional
            If True, fits an intercept in the lag model. Default is False.
        cv : int, optional
            Number of cross-validation folds to use when selecting lags. Default is 5.
        shuffle : bool, optional
            If True, shuffle the data during cross-validation. Default is True.
        n_iter : int, optional
            Number of iterations to use during randomized cross-validation or lag selection. Default is 20.
        var_order : str, optional
            The order in which variables are considered for lag selection. Options include 'cross' (cross-sectional ordering) and others. Default is 'cross'.
        max_lag : int, optional
            The maximum lag to consider for the model. Default is 3.
        y_lags : str or int, optional
            The number of lags for the target variable (`y`). If 'Auto', the lag length is automatically determined. Default is 'Auto'.
        exog_lags : str or int, optional
            The number of lags for the exogenous variables (`exog`). If 'Auto', the lag length is automatically determined. Default is 'Auto'.

        Returns
        -------
        None
            Updates the lag-related attributes of the instance, such as `lag_select`, `lag_fit_intercept`, `lag_cv`, and others.

        Notes
        -----
        - This method sets the parameters for the lag selection process, which determines the temporal dependencies in the model.
        - The selection can be done using either information criteria or cross-validation approaches.
        - The `var_order` parameter controls the order in which variables are processed for lag determination, which can impact model performance.
        """
        
        self.lag_select = lag_select
        self.lag_fit_intercept = fit_intercept
        self.lag_cv = cv
        self.var_order = var_order
        self.max_lag = max_lag
        self.y_lags = y_lags
        self.exog_lags = exog_lags
        self.shuffle = shuffle
        self.lag_iter = n_iter
    
    def forecast(self, h_list=[1,3,6,12], seasonal=False, hsi_CV_select=False, fit_intercept=False, cv=4, n_iter=100, original_scale=True, sort_df='criteria',  sort_col=-1):
        """
        Forecasts the Housing Price Index (HPI) using prepared Google Trends data and selected lag features.
        
        Parameters
        ----------
        h_list : list of int, optional
            A list of forecast horizons for which the predictions will be made. Default is [1, 3, 6, 12].
        seasonal : bool, optional
            If True, includes seasonal components in the model during forecasting. Default is False.
        hsi_CV_select : bool, optional
            If True, uses cross-validation for selecting the Housing Sentiment Index (HSI) features. Default is False.
        fit_intercept : bool, optional
            If True, fits an intercept term in the forecasting model. Default is False.
        cv : int, optional
            Number of cross-validation folds for model validation. Default is 4.
        n_iter : int, optional
            Number of iterations to use during cross-validation or parameter tuning. Default is 100.
        original_scale : bool, optional
            If True, the forecasted values are converted back to their original scale. Default is True.
        sort_df : str, optional
            The method used to sort the features for the forecast model. Options include 'criteria'. Default is 'criteria'.
        sort_col : int, optional
            The column index to use for sorting when selecting features. Default is -1 (the last column).

        Returns
        -------
        None
            Updates the attributes with forecast results, including various performance metrics and forecast data.

        Attributes Updated
        ------------------
        MAFE_val_df : pd.DataFrame
            Mean Absolute Forecasting Error (MAFE) values for the validation set.
        MAFE_val_df_improve : pd.DataFrame
            Improvement in MAFE values compared to a benchmark.
        MSFE_val_df : pd.DataFrame
            Mean Squared Forecasting Error (MSFE) values for the validation set.
        MSFE_val_df_improve : pd.DataFrame
            Improvement in MSFE values compared to a benchmark.
        forecast_criteria_df : pd.DataFrame
            DataFrame containing criteria values used for selecting features during forecasting.
        lags_df : pd.DataFrame
            DataFrame representing the lags used for each feature in the model.
        MAFE_train_df : pd.DataFrame
            MAFE values for the training set.
        MSFE_train_df : pd.DataFrame
            MSFE values for the training set.
        forecast_dict : dict
            Dictionary containing forecasted values for different horizons.
        hsi : pd.DataFrame
            DataFrame containing the forecasted HPI values.
        results_table : pd.DataFrame
            Table summarizing the forecast results, including performance metrics.

        Notes
        -----
        - This method forecasts the HPI using a combination of Google Trends data and selected lagged features.
        - Various forecast horizons can be specified using the `h_list` parameter to produce forecasts for different time periods.
        - The method can optionally fit an intercept term and use cross-validation to select the most important features.
        """
        
        self.h_list = h_list
        self.original_scale = original_scale
        
        MAFE_val_df, MAFE_val_df_improve, MSFE_val_df, MSFE_val_df_improve, forecast_criteria_df, lags_df, MAFE_train_df, MSFE_train_df, forecast_dict, df = compare_exog(self.forecast_hsi, train_cut=self.train_cut, h_list=h_list, seasonal=seasonal, max_lag=self.max_lag, lag_select=self.lag_select, seed=self.seed, lag_fit_intercept=self.lag_fit_intercept, lag_cv=self.lag_cv, lag_shuffle=self.shuffle, lag_iter=self.lag_iter, hsi_CV_select=hsi_CV_select, hsi_fit_intercept=fit_intercept, hsi_cv=cv, hsi_iter=n_iter, y_lags=self.y_lags, exog_lags=self.exog_lags, var_order=self.var_order, verbose=self.verbose, log=self.log, original_hpi=self.original_hpi, original_scale=original_scale, sort_df=sort_df,  sort_col=sort_col)
        self.MAFE_val_df = MAFE_val_df
        self.MAFE_val_df_improve = MAFE_val_df_improve
        self.MSFE_val_df = MSFE_val_df
        self.MSFE_val_df_improve = MSFE_val_df_improve
        self.forecast_criteria_df = forecast_criteria_df
        self.lags_df = lags_df
        self.MAFE_train_df = MAFE_train_df
        self.MSFE_train_df = MSFE_train_df
        self.forecast_dict = forecast_dict
        self.hsi = df
        self.results_table = forecast_table(self)
        # self.hsi_dfs[-1] = df
        
    def results(self, results_table=False, MAFE_val_df=False, MAFE_val_df_improve=False, MSFE_val_df=False, MSFE_val_df_improve=False, forecast_criteria_df=False, lags_df=False, MAFE_train_df=False, head=10):
        """
        Displays various results from the model, including performance metrics and forecast data.

        Parameters
        ----------
        results_table : bool, optional
            If True, displays the full results table summarizing the forecast performance. Default is False.
        MAFE_val_df : bool, optional
            If True, displays the Mean Absolute Forecasting Error (MAFE) values for the validation set. Default is False.
        MAFE_val_df_improve : bool, optional
            If True, displays the improvement in MAFE values compared to a benchmark. Default is False.
        MSFE_val_df : bool, optional
            If True, displays the Mean Squared Forecasting Error (MSFE) values for the validation set. Default is False.
        MSFE_val_df_improve : bool, optional
            If True, displays the improvement in MSFE values compared to a benchmark. Default is False.
        forecast_criteria_df : bool, optional
            If True, displays the DataFrame containing criteria values used for selecting features during forecasting. Default is False.
        lags_df : bool, optional
            If True, displays the DataFrame representing the lags used for each feature in the model. Default is False.
        MAFE_train_df : bool, optional
            If True, displays the MAFE values for the training set. Default is False.
        head : int, optional
            Number of rows to display when displaying the results DataFrames. Default is 10.

        Returns
        -------
        None
            Displays the specified results based on the provided parameters.

        Notes
        -----
        - This method allows the user to access different performance metrics and forecast-related data.
        - By selecting the appropriate parameters, users can view specific tables and metrics that summarize the model's forecasting performance.
        - The `head` parameter controls the number of rows to display for DataFrames to avoid overwhelming output.
        """

        results(self, results_table=results_table, MAFE_val_df=MAFE_val_df, MAFE_val_df_improve=MAFE_val_df_improve, MSFE_val_df=MSFE_val_df, MSFE_val_df_improve=MSFE_val_df_improve, forecast_criteria_df=forecast_criteria_df, lags_df=lags_df, MAFE_train_df=MAFE_train_df, head=head)
    
    def plot_gtrends(self, exog=None, h=12, winsorize_trend=False, scaled=False):
        """
        Parameters
        ----------
        exog : list of str or None, optional
            List of exogenous variables (Google Trends features) to plot. If None, all available features are plotted. Default is None.
        h : int, optional
            The forecast horizon for which the trends are plotted. Default is 12.
        winsorize_trend : bool, optional
            If True, plots the Google Trends data after winsorization, which reduces the effect of outliers. Default is False.
        scaled : bool, optional
            If True, plots the scaled version of Google Trends data, allowing for comparison across different features. Default is False.

        Returns
        -------
        None
            Displays the plots for the specified Google Trends features.

        Notes
        -----
        - This method provides a visual representation of Google Trends data used in the model, which helps in understanding the temporal patterns in the data.
        - Users can optionally plot specific features by providing a list in the `exog` parameter.
        - The `winsorize_trend` parameter allows for a clearer view by minimizing the impact of extreme values.
        - The `scaled` parameter enables visualization of standardized features, useful for comparing different variables on the same scale.
        """
        
        plot_gtrends(self, exog=exog, h=h, winsorize_trend=winsorize_trend, scaled=scaled)
        
    def plot_hsi(self, hsi=None):
        """
        Plots the Housing Sentiment Index (HSI) data used in the model, allowing for visualization of the extracted features.

        Parameters
        ----------
        hsi : list of str or None, optional
            List of HSI features to plot. If None, all available HSI features are plotted. Default is None.

        Returns
        -------
        None
            Displays the plots for the specified HSI features.

        Notes
        -----
        - This method provides a visual representation of the HSI features that have been extracted using PCA and other transformations.
        - The `hsi` parameter allows users to specify particular HSI features to visualize, or plot all available features if set to None.
        - Useful for analyzing the derived sentiment features and understanding their temporal patterns.
        """
        plot_hsi(self, hsi=hsi)
        
    def plot_improvement(self, measure='MAFE', h_list=None, colors=None, labels=None, figsize=(15,9)):
        """ 
        Plots the improvement in forecasting accuracy for different horizons, using a specified performance measure.

        Parameters
        ----------
        measure : str, optional
            The performance measure to visualize. Options include 'MAFE' (Mean Absolute Forecasting Error) and 'MSFE' (Mean Squared Forecasting Error). Default is 'MAFE'.
        h_list : list of int or None, optional
            A list of forecast horizons to include in the plot. If None, all available horizons are included. Default is None.
        colors : list of str or None, optional
            List of colors to use for the plot lines. If None, default colors are used. Default is None.
        labels : list of str or None, optional
            List of labels for each forecast horizon in the plot. If None, default labels are used. Default is None.
        figsize : tuple of int, optional
            Figure size for the plot. Default is (15, 9).

        Returns
        -------
        None
            Displays a plot showing the improvement in forecasting accuracy for the specified measure and horizons.

        Notes
        -----
        - This method helps visualize the improvement in forecasting performance for different horizons.
        - The `measure` parameter allows the user to specify whether to visualize Mean Absolute Forecasting Error (MAFE) or Mean Squared Forecasting Error (MSFE).
        - Users can customize the appearance of the plot using the `colors`, `labels`, and `figsize` parameters.
        - The `h_list` parameter allows for selecting specific forecast horizons to analyze, which can be useful for evaluating the model's performance over varying time periods.
        """
        
        plot_improvement([self], measure=measure, h_list=h_list, colors=colors, labels=labels, figsize=figsize)
        
    def plot_forecast(self, exog=None):
        """
        Plots the forecasted Housing Price Index (HPI) values along with the actual values for comparison.
        
        Parameters
        ----------
        exog : list of str or None, optional
            List of exogenous variables (features) to include in the forecast plot. If None, all available features are included. Default is None.

        Returns
        -------
        None
            Displays the forecast plot showing the predicted HPI values and the actual values.

        Notes
        -----
        - This method provides a visual comparison between the forecasted HPI values and the actual values, allowing users to assess the model's predictive performance.
        - The `exog` parameter allows users to include specific exogenous features in the plot for further analysis of their impact on the forecast.
        - Useful for evaluating the quality of the model's predictions over different time horizons.
        """
        
        plot_forecast(self, exog=exog)