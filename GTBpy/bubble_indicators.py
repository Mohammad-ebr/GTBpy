"""
Bubble tests from the following papers:
- PWY (SADF test): Phillips, Wu and Yu (2011)
- PSY (GSADF test): Phillips, Shi and Yu (2015)
- WHL: Whitehouse, Harvey and Leybourne (2022)
"""

from numba import njit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


#%%
def ADF_FL(y, adflag, regression='c'):
    return adfuller(y, maxlag=adflag, regression=regression, autolag=None)[0]

#%%
@njit
def ADF_FL_njit(y, adflag, regression='c'):
    y = y.flatten().astype(np.float64)
    t1 = len(y) - 1
    const = np.ones((t1,1))
    trend = np.arange(1, t1 + 1).reshape(-1, 1)
    
    x = y[:len(y)-1].reshape(-1, 1)
    dy = np.diff(y, n=1)
    
    if regression == 'c':
        x = np.column_stack((x, const))
    elif regression == 'ct':
        x = np.column_stack((x, const, trend))
    
    t2 = t1 - adflag
    x2 = x[len(x)-t2:]
    dy01 = dy[len(dy)-t2:]
    
    if adflag > 0:
        for j in range(1, adflag + 1):
            x2 = np.column_stack((x2, dy[len(dy) -t2 -j : len(dy) -j]))
    
    beta = np.linalg.inv(x2.T @ x2) @ (x2.T @ dy01)
    eps = dy01 - x2 @ beta
    
    if regression == 'n':
        k = 1
    elif regression == 'c':
        k = 2
    elif regression == 'ct':
        k = 3
    
    sig = np.sqrt(np.diag((eps.T @ eps) / (t2 - adflag - k) * np.linalg.inv(x2.T @ x2)))
    tvalue = beta / sig

    return tvalue[0]

#%%
def bubbles(stat, cv):
    # Boolean mask where Stat > Critical value
    mask = stat > cv

    # Find periods where mask is True
    periods = []
    start_date = None
    
    for i in range(len(mask)):
        current_date = stat.index[i]
        
        if mask[i] and start_date is None:
            # Start of a new period
            start_date = current_date
        elif not mask[i] and start_date is not None:
            # End of the current period
            periods.append([start_date, current_date])
            start_date = None
        elif i == len(mask) - 1 and mask[i]:
            # Special case for when the period extends to the last row
            periods.append([start_date, current_date])
    
    # Create the result DataFrame with the periods
    bubbles = pd.DataFrame(periods, index=[f'bubble_{i}' for i in range(1, len(periods)+1)], columns=['Start Date', 'End Date'])
    
    return bubbles

#%%
def plot(self, stat, cv, label):
    fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,9))
    axs[0].plot(self.y, color='g', label='y')
    axs[1].plot(stat, color='b', label=label)
    axs[1].plot(cv[self.test_qe], color='r', label=f'{label} {self.test_qe} critical values') #  marker='.',
    
    for ax in axs:
        ax.legend()
        for idx in self.bubbles_df.index:
            ax.axvspan(self.bubbles_df.loc[idx, 'Start Date'], self.bubbles_df.loc[idx, 'End Date'], color='gray', alpha=0.2)
            
#%%
class PWY_SADF():
    """
    Implements the Supremum Augmented Dickey-Fuller (SADF) test, based on Phillips, Wu, and Yu (2011),
    to detect and date-stamp speculative bubbles in time series data.

    The PWY SADF test is an extension of the ADF test, where recursive right-tailed ADF tests are performed 
    on expanding subsamples of the data. The test identifies periods when the test statistic exceeds a critical 
    value, which can be interpreted as evidence of bubble-like behavior.

    Parameters
    ----------
    y : pandas.Series, optional
        Time series data on which the SADF test will be performed.
    swindow0 : int, optional
        Minimum window size for the recursive ADF tests. If not provided, it defaults to 
        `int(T * (0.01 + 1.8 / np.sqrt(T)))`, where `T` is the length of `y`.
    adflag : int, default=1
        The lag length for the ADF test.
    regression : {'c', 'ct', 'n'}, default='c'
        Type of deterministic term included in the ADF test:
        - 'c' : constant.
        - 'ct': constant and trend.
        - 'n' : no deterministic terms.
    njit : bool, default=False
        If True, uses a Numba-optimized version of the ADF test to speed up computation.

    Attributes
    ----------
    y : pandas.Series
        The input time series data.
    T : int
        Length of the input time series.
    swindow0 : int
        The minimum window size used in recursive ADF tests.
    adflag : int
        The number of lags used in the ADF test.
    regression : str
        Type of regression term used in the ADF test ('c', 'ct', or 'n').
    ADF : callable
        The function used to compute the ADF statistic. Either `ADF_FL` or `ADF_FL_njit` depending 
        on the `njit` parameter.
    sadf : float
        The supremum ADF statistic computed from the recursive ADF tests.
    badfs : pandas.Series
        The ADF statistics computed for each subsample, indexed by time.
    cv_sadf : numpy.ndarray
        Critical values for the SADF statistic at specified quantiles.
    cv_badfs : pandas.DataFrame
        Critical values for the badfs statistics at specified quantiles.
    bubbles_df : pandas.DataFrame
        Identified bubble periods, with start and end dates for each bubble.
    test_qe : float
        Quantile used to test for bubbles in the `bubbles` method.

    Methods
    -------
    stats()
        Computes the SADF statistic and the ADF statistics for each subsample.
    critical_values(T=None, cv_qes=[0.9, 0.95, 0.99], m=2000, seed=1)
        Simulates critical values for the SADF and badfs statistics based on Monte Carlo simulations.
    bubbles(test_qe)
        Identifies bubble periods where the SADF statistic exceeds the critical values for the given quantile.
    plot()
        Plots the time series, SADF statistics, critical values, and highlights bubble periods.

    References
    ----------
    Phillips, P. C., Wu, Y., & Yu, J. (2011). Explosive behavior in the 1990s NASDAQ: When did exuberance escalate asset values?.
    International Economic Review, 52(1), 201-226.

    Examples
    --------
    >>> pw_sadf = PWY_SADF(y=price_series, swindow0=30, adflag=1, regression='c')
    >>> sadf_stat, badfs_stats = pw_sadf.stats()  # Compute the SADF statistics
    >>> cv_sadf, cv_badfs = pw_sadf.critical_values(cv_qes=[0.95, 0.99], m=1000, seed=42)  # Simulate critical values
    >>> bubbles_df = pw_sadf.bubbles(test_qe=0.95)  # Identify bubble periods
    >>> pw_sadf.plot()  # Plot the results
    """
    
    def __init__(self, y=None, swindow0=None, adflag=1, regression='c', njit=False) -> None:
        
        self.y = y
        self.T = len(y) if y is not None else None
        self.swindow0 = swindow0 if swindow0 else int(self.T * (0.01 + 1.8 / np.sqrt(self.T)))
        self.adflag = adflag
        self.regression = regression
        self.ADF = ADF_FL_njit if njit else ADF_FL
    
    def stats(self):
        """
        Calculates the Supremum Augmented Dickey-Fuller (SADF) test statistic over expanding windows.

        This method performs a recursive ADF test using expanding sample windows starting from the 
        initial window size `swindow0` up to the total length of the time series `T`. For each window, 
        the ADF statistic is computed, and the SADF statistic is determined as the maximum value across 
        all ADF statistics.

        Returns
        -------
        sadf : float
            The supremum ADF (SADF) statistic, which is the maximum of the ADF statistics over all expanding windows.
        
        badfs : pd.Series
            A Pandas Series containing the ADF statistics for each window size from `swindow0` to `T`. 
            The index of the Series corresponds to the time index of the input series `y`.

        Notes
        -----
        - The ADF statistic is calculated using the specified `adflag` (number of lags) and `regression`
          (constant or constant with trend).
        - The SADF test is a recursive right-tailed unit root test, commonly used for bubble detection in
          time series data.
        - The `badfs` series represents the sequence of ADF test statistics for each expanding window,
          where the length of the window increases from `swindow0` to `T`.

        Examples
        --------
        >>> pw_sadf = PWY_SADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> sadf_stat, adf_stats = pw_sadf.stats()
        >>> print(sadf_stat)
        2.345  # Example output for the maximum ADF statistic
        >>> print(adf_stats.head())
        2000-01-31    NaN
        2000-02-29    NaN
        2000-03-31    NaN
        2000-04-30    NaN
        2000-05-31   -1.23
        dtype: float64
        """

        badfs = np.full(self.T, np.nan)
        y_array = np.array(self.y) # Convert y series to array, in order to speed up the calculation by @njit
        for i in range(self.swindow0, self.T + 1):
            badfs[i - 1] = self.ADF(y_array[:i], adflag=self.adflag, regression=self.regression)
            
        self.sadf = np.max(badfs)
        self.badfs = pd.Series(badfs, index=self.y.index)
    
        return self.sadf, self.badfs
    
    
    def critical_values(self, T=None, cv_qes=[0.9, 0.95, 0.99], m=2000, seed=1):
        """
        Computes the critical values for the Supremum Augmented Dickey-Fuller (SADF) test.

        This method simulates random walks under the null hypothesis of a unit root and calculates the 
        SADF statistics for each simulation. Based on these simulations, it derives the critical values 
        for the SADF test at specified quantiles. It also provides the corresponding critical value 
        series for each window in the expanding sequence.

        Parameters
        ----------
        T : int, optional
            The sample size used in the simulation. If not provided, the sample size of the observed time 
            series `y` is used. Defaults to `None`.
        
        cv_qes : list of float, optional
            Quantiles for which the critical values should be calculated. These quantiles represent the 
            desired significance levels. The default values are [0.9, 0.95, 0.99], corresponding to the 
            10%, 5%, and 1% significance levels, respectively.
        
        m : int, optional
            The number of Monte Carlo simulations used to generate the critical values. Defaults to 2000.
        
        seed : int, optional
            Random seed used to ensure reproducibility of the simulations. Defaults to 1.

        Returns
        -------
        cv_sadf : np.ndarray
            An array containing the SADF critical values at the specified quantiles (e.g., 90th, 95th, and 99th percentiles).
        
        cv_badfs : pd.DataFrame
            A DataFrame where each column corresponds to the critical value series for a specific quantile, 
            indexed by the time series index of `y` (if `T == len(y)`). Each column contains the critical 
            values for the SADF test at the respective quantile for each window size.

        Notes
        -----
        - The method generates `m` random walk series and computes the SADF statistics for each of them.
        - The critical values are derived by taking the quantiles of the SADF statistics across the
          simulations.
        - The `cv_badfs` DataFrame provides the critical values at each window size, which can be used
          to compare with the actual SADF statistics obtained from the observed data.

        Examples
        --------
        >>> pw_sadf = PWY_SADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> sadf_cv, adf_cv_series = pw_sadf.critical_values(cv_qes=[0.95, 0.99], m=1000, seed=42)
        >>> print(sadf_cv)
        [1.750, 2.100]  # Example critical values at 95% and 99% levels
        >>> print(adf_cv_series.head())
                    0.95   0.99
        2000-01-31    NaN    NaN
        2000-02-29    NaN    NaN
        2000-03-31    NaN    NaN
        2000-04-30    NaN    NaN
        2000-05-31   1.23   1.75
        dtype: float64
        """

        if T is None:
            T = self.T
            
        # Data generating process
        dim = self.T - self.swindow0 + 1
        np.random.seed(seed)
        e = np.random.randn(self.T, m)
        a = self.T ** (-1)
        y = np.cumsum(e + a, axis=0)
        
        # Sup ADF test
        badfs = np.full((m, dim), np.nan)
        sadf = np.full(m, np.nan)
        
        for j in range(m):
            for i in range(self.swindow0, self.T + 1):
                badfs[j, i - self.swindow0] = self.ADF(y[:i, j], adflag=self.adflag, regression=self.regression)
        
        sadf = np.max(badfs, axis=1)
        
        self.cv_sadf = np.quantile(sadf, cv_qes)
        cv_badfs = np.vstack((np.full(((self.T - dim), len(cv_qes)), np.nan), np.quantile(badfs, cv_qes, axis=0).T))
        self.cv_badfs = pd.DataFrame(cv_badfs, columns=cv_qes)
        # if (self.y is not None) and len(self.y) == len(self.cv_badfs):
        if T == self.T:
            self.cv_badfs.index = self.y.index
        
        return self.cv_sadf, self.cv_badfs

    def bubbles(self, test_qe):
        """
        Identifies periods of explosive behavior (bubbles) based on the Supremum Augmented Dickey-Fuller (SADF) test results.

        This method compares the SADF statistics (badfs) with the critical value series at a specified quantile 
        and identifies time periods where the SADF statistic exceeds the critical value. These periods are 
        considered as potential bubbles.

        Parameters
        ----------
        test_qe : float
            The quantile used to select the critical value series for comparison. This quantile should be one of the 
            values used in the `critical_values` method (e.g., 0.9, 0.95, 0.99) to determine the significance level.

        Returns
        -------
        bubbles_df : pd.DataFrame
            A DataFrame containing the start and end dates of the identified bubbles. Each row represents a separate 
            bubble episode, and the DataFrame has two columns:
            - 'Start Date': The date when the bubble period begins (when the SADF statistic first exceeds the critical value).
            - 'End Date': The date when the bubble period ends (when the SADF statistic falls back below the critical value).

        Notes
        -----
        - The method relies on the comparison of the SADF statistic (`badfs`) and the critical value series at the specified quantile.
        - Consecutive periods where the SADF statistic exceeds the critical value are considered part of the same bubble.
        - The identified bubble periods are returned as a DataFrame, which can be useful for further analysis or visualization.

        Examples
        --------
        >>> pw_sadf = PWY_SADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> pw_sadf.stats()  # Compute the SADF statistics
        >>> pw_sadf.critical_values(cv_qes=[0.95, 0.99], m=1000, seed=42)  # Compute critical values
        >>> bubbles_df = pw_sadf.bubbles(test_qe=0.95)  # Identify bubble periods using the 95% critical value
        >>> print(bubbles_df)
                    Start Date   End Date
        bubble_1    2002-03-31  2002-07-31
        bubble_2    2004-05-31  2004-09-30
        bubble_3    2007-01-31  2007-04-30
        """
        self.test_qe = test_qe
        self.bubbles_df = bubbles(self.badfs, self.cv_badfs[test_qe])
        return self.bubbles_df

    def plot(self):
        """
        Plots the time series of the input data along with the SADF statistics and critical values, highlighting identified bubble periods.

        This method generates two subplots:
        - The first plot shows the time series of the input data (`y`).
        - The second plot shows the SADF statistics (`badfs`) and the corresponding critical values. 
        Periods identified as bubbles are shaded in both plots.

        The method visually displays the relationship between the SADF statistics and critical values to help 
        interpret when and where bubble periods occur.

        Returns
        -------
        None
            The method generates a plot but does not return any objects.

        Notes
        -----
        - Bubble periods are shaded in gray on both the input data and SADF statistic plots.
        - Critical values are shown in red, and the SADF statistic is shown in blue.
        - The method assumes that the `bubbles` method has already been called to identify bubble periods
          and populate the `bubbles_df` attribute.

        Examples
        --------
        >>> pw_sadf = PWY_SADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> pw_sadf.stats()  # Compute the SADF statistics
        >>> pw_sadf.critical_values(cv_qes=[0.95, 0.99], m=1000, seed=42)  # Compute critical values
        >>> pw_sadf.bubbles(test_qe=0.95)  # Identify bubble periods
        >>> pw_sadf.plot()  # Plot the data, SADF statistics, and bubble periods

        The resulting plot will show:
        - The price series in green (first subplot).
        - The SADF statistics in blue and critical values in red (second subplot).
        - Gray-shaded areas indicating identified bubble periods on both subplots.
        """
        plot(self, stat=self.badfs, cv=self.cv_badfs, label='badfs')

#%%
class PSY_GSADF():
    """
    Implements the Generalized Supremum Augmented Dickey-Fuller (GSADF) test, based on Phillips, Shi, 
    and Yu (2015), to detect and date-stamp multiple speculative bubbles in a time series.

    The GSADF test extends the SADF test by allowing both the start and end points of the sample 
    window to vary, making it more robust in detecting multiple periods of explosive behavior. This 
    test is widely used for identifying bubbles in financial and economic time series.

    Parameters
    ----------
    y : pandas.Series, optional
        The time series data on which the GSADF test will be performed.
    swindow0 : int, optional
        The minimum window size for the backward SADF (BSADF) calculations. If not provided, it defaults 
        to `int(T * (0.01 + 1.8 / np.sqrt(T)))`, where `T` is the length of the time series.
    adflag : int, default=1
        The number of lags used in the ADF test.
    regression : {'c', 'ct', 'n'}, default='c'
        The type of deterministic term used in the ADF test:
        - 'c' : constant.
        - 'ct': constant and trend.
        - 'n' : no deterministic terms.
    njit : bool, default=False
        If True, the method uses a Numba-optimized version of the ADF test to improve computational efficiency.

    Attributes
    ----------
    y : pandas.Series
        The input time series data.
    T : int
        The length of the input time series.
    swindow0 : int
        The minimum window size used for backward SADF calculations.
    adflag : int
        The number of lags used in the ADF test.
    regression : str
        The type of deterministic term used in the ADF test ('c', 'ct', or 'n').
    ADF : callable
        The function used to compute the ADF statistic, either `ADF_FL` or `ADF_FL_njit` depending on the `njit` parameter.
    gsadf : float
        The Generalized Supremum ADF (GSADF) statistic, representing the maximum BSADF statistic across all windows.
    bsadfs : pandas.Series
        A series of BSADF statistics, indexed by the time series' index.
    cv_gsadf : numpy.ndarray
        Critical values for the GSADF statistic at specified quantiles.
    cv_bsadfs : pandas.DataFrame
        Critical values for the BSADF statistics over time at specified quantiles.
    bubbles_df : pandas.DataFrame
        Detected bubble periods with their start and end dates.
    test_qe : float
        The quantile used to test for bubbles in the `bubbles()` method.

    Methods
    -------
    stats()
        Computes the GSADF statistic and the BSADF statistics for varying windows.
    critical_values(T=None, bsadfs=None, cv_qes=[0.9, 0.95, 0.99], m=2000, seed=1)
        Computes the critical values for the GSADF and BSADF statistics using Monte Carlo simulations.
    bubbles(test_qe)
        Identifies bubble periods by comparing the BSADF statistics to the critical values at a given quantile.
    plot()
        Plots the time series, BSADF statistics, and critical values, highlighting bubble periods.

    References
    ----------
    Phillips, P. C., Shi, S., & Yu, J. (2015). Testing for multiple bubbles: Historical episodes of exuberance and 
    collapse in the S&P 500. *International Economic Review*, 56(4), 1043-1078.

    Examples
    --------
    >>> gsadf_test = PSY_GSADF(y=price_series, swindow0=30, adflag=1, regression='c')
    >>> gsadf_stat, bsadfs_stats = gsadf_test.stats()  # Compute GSADF and BSADF statistics
    >>> cv_gsadf, cv_bsadfs = gsadf_test.critical_values(m=5000, cv_qes=[0.9, 0.95, 0.99])
    >>> bubbles_df = gsadf_test.bubbles(test_qe=0.99)  # Detect bubble periods
    >>> gsadf_test.plot()  # Visualize the results
    """
    
    def __init__(self, y=None, swindow0=None, adflag=1, regression='c', njit=False) -> None:
        
        self.y = y
        self.T = len(y) if y is not None else None
        self.swindow0 = swindow0 if swindow0 else int(self.T * (0.01 + 1.8 / np.sqrt(self.T)))
        self.adflag = adflag
        self.regression = regression
        self.ADF = ADF_FL_njit if njit else ADF_FL
        
            
    def stats(self):
        """
        Computes the Generalized Supremum Augmented Dickey-Fuller (GSADF) statistic and the backward SADF 
        statistics for expanding windows of the time series.

        The GSADF test is an extension of the SADF test, where both the start and end points of the estimation 
        window vary, making the test more robust in detecting multiple bubbles. This method computes the GSADF 
        statistic, which is the maximum backward SADF statistic over the sample.

        Returns
        -------
        gsadf : float
            The Generalized Supremum ADF (GSADF) statistic, which is the maximum backward SADF statistic over 
            the expanding sample.
        bsadfs : pandas.Series
            A time series of backward SADF (BSADF) statistics, calculated for each window endpoint.

        Notes
        -----
        The GSADF test is useful in identifying speculative bubbles by testing for explosive behavior in multiple 
        periods, making it more sensitive to detecting both the origination and collapse of bubbles.

        References
        ----------
        Phillips, P. C., Shi, S., & Yu, J. (2015). Testing for multiple bubbles: Historical episodes of exuberance and collapse 
        in the S&P 500. *International Economic Review*, 56(4), 1043-1078.

        Examples
        --------
        >>> gsadf_test = PSY_GSADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> gsadf_stat, bsadfs_stats = gsadf_test.stats()  # Compute the GSADF and BSADF statistics
        """
        bsadfs = np.full(self.T, np.nan)
        y_array = np.array(self.y)
        for r2 in range(self.swindow0, self.T + 1):
            dim0 = r2 - self.swindow0 + 1
            rwadft = np.zeros(dim0)
            for r1 in range(1, dim0 + 1):
                rwadft[r1 - 1] = self.ADF(y_array[r1-1:r2], 0, 1)
            bsadfs[r2 - 1] = np.max(rwadft)
            
        self.gsadf = np.max(bsadfs)
        self.bsadfs = pd.Series(bsadfs, index=self.y.index)
        
        return self.gsadf, self.bsadfs
    
    def critical_values(self, T=None, bsadfs=None, cv_qes=[0.9, 0.95, 0.99], m=2000, seed=1):
        """
        Computes the critical values for the Generalized Supremum Augmented Dickey-Fuller (GSADF) test 
        based on Monte Carlo simulations.

        Parameters
        ----------
        T : int, optional
            Length of the time series. If not provided, the length of the series (`self.T`) is used.
        bsadfs : numpy.ndarray, optional
            Precomputed matrix of backward SADF (BSADF) statistics from previous simulations, 
            allowing continuation of simulations. If None, a new matrix is initialized.
        cv_qes : list of float, optional
            Quantiles for which critical values are computed, by default [0.9, 0.95, 0.99].
        m : int, optional
            Number of Monte Carlo simulations to run for generating the critical values, by default 2000.
        seed : int, optional
            Seed for the random number generator used in Monte Carlo simulations, by default 1.

        Returns
        -------
        cv_gsadf : numpy.ndarray
            Critical values for the GSADF test at the specified quantiles.
        cv_bsadfs : pandas.DataFrame
            Critical values for the BSADF statistics over time at the specified quantiles.

        Notes
        -----
        The critical values are obtained by simulating a random walk process and computing the GSADF 
        and BSADF statistics for each simulation. These statistics are then used to calculate the 
        empirical quantiles, which serve as the critical values.

        The random walk is generated as a cumulative sum of standard normal shocks, adjusted with a 
        small drift term (`a = T**(-1)`), where `T` is the length of the time series.

        Examples
        --------
        >>> gsadf_test = PSY_GSADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> cv_gsadf, cv_bsadfs = gsadf_test.critical_values(m=5000, cv_qes=[0.9, 0.95, 0.99])
        """
        if T is None:
            T = self.T
        # Data generating process
        dim = T - self.swindow0 + 1
        np.random.seed(seed)
        e = np.random.randn(T, m)
        a = T ** (-1)
        y = np.cumsum(e + a, axis=0)
        
        # Generalized sup ADF test
        gsadf = np.ones(m)
        if bsadfs is None:
            start = 0
            bsadfs = np.zeros((m, dim))
        else:
            start = next((i for i, row in enumerate(bsadfs) if np.isnan(row).any()), m-1)
        
        for j in range(start, m):
            for r2 in range(self.swindow0, T + 1):
                dim0 = r2 - self.swindow0 + 1
                rwadft = np.zeros(dim0)
                for r1 in range(1, dim0 + 1):
                    rwadft[r1 - 1] = self.ADF(y[r1-1:r2, j], 0, 1)
                bsadfs[j, r2 - self.swindow0] = np.max(rwadft)
            
        gsadf = np.max(bsadfs, axis=1)
        self.cv_gsadf = np.quantile(gsadf, cv_qes)
        cv_bsadfs = np.vstack((np.full(((self.T - dim), len(cv_qes)), np.nan), np.quantile(bsadfs, cv_qes, axis=0).T))
        self.cv_bsadfs = pd.DataFrame(cv_bsadfs, columns=cv_qes)
        if self.T == T:
            self.cv_bsadfs.index = self.y.index
        
        return self.cv_gsadf, self.cv_bsadfs
    
    def bubbles(self, test_qe):
        """
        Identifies and returns periods of explosive behavior (bubbles) in the time series 
        based on the GSADF test statistic and its critical values.

        Parameters
        ----------
        test_qe : float
            The quantile level (e.g., 0.95 or 0.99) of the critical value to use for bubble identification. 
            It must match one of the quantiles used in `critical_values()`.

        Returns
        -------
        bubbles_df : pandas.DataFrame
            DataFrame containing the start and end dates of detected bubble periods.
            The index is labeled as 'bubble_1', 'bubble_2', etc., for each identified bubble.
            The columns are:
            - 'Start Date': The start date of the bubble.
            - 'End Date': The end date of the bubble.

        Notes
        -----
        A bubble is detected when the BSADF statistic exceeds the critical value corresponding 
        to the specified quantile level (`test_qe`). The method identifies periods during which 
        the time series shows signs of explosive growth.

        The method applies a rolling window approach where at each time step the BSADF statistic 
        is compared to the critical value. If the statistic is greater than the critical value, 
        it indicates a bubble, and the corresponding time period is recorded.

        Examples
        --------
        >>> gsadf_test = PSY_GSADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> gsadf_test.critical_values(m=5000, cv_qes=[0.9, 0.95, 0.99])
        >>> bubbles_df = gsadf_test.bubbles(test_qe=0.99)
        >>> print(bubbles_df)
        """
        self.test_qe = test_qe
        self.bubbles_df = bubbles(self.bsadfs, self.cv_bsadfs[test_qe])
        return self.bubbles_df

    def plot(self):
        """
        Plots the time series and the corresponding BSADF test statistics along with the critical values,
        highlighting the periods of explosive behavior (bubbles).

        Parameters
        ----------
        None

        Returns
        -------
        None
            Displays a plot with two subplots:
            - The upper subplot shows the original time series (`y`).
            - The lower subplot shows the BSADF test statistics and the corresponding critical values 
            for a given quantile level (`test_qe`).
            Detected bubble periods are shaded in both plots.

        Notes
        -----
        The method visualizes the results of the GSADF test by plotting the following:
        1. The original time series (`y`) in the upper plot.
        2. The BSADF statistics in the lower plot, where they are compared against the critical values 
        (computed in the `critical_values()` method) at the specified quantile level (`test_qe`).
        3. Periods identified as bubbles (from the `bubbles()` method) are shaded in both plots 
        for easy identification.

        Examples
        --------
        >>> gsadf_test = PSY_GSADF(y=price_series, swindow0=30, adflag=1, regression='c')
        >>> gsadf_test.critical_values(m=5000, cv_qes=[0.9, 0.95, 0.99])
        >>> gsadf_test.bubbles(test_qe=0.99)
        >>> gsadf_test.plot()
        """
        plot(self, stat=self.bsadfs, cv=self.cv_bsadfs, label='bsadfs')

#%%
class WHL():
    """
    Implements the Whitehouse, Harvey, and Leybourne (2022) test for detecting and analyzing 
    speculative bubbles and crashes in time series data.

    This class calculates two key statistics: 
    - **A-statistic**: Used to detect the presence of bubbles (explosive behavior).
    - **S-statistic**: Used to identify crashes (sharp declines) following bubbles.

    Parameters
    ----------
    y : pandas.Series
        The time series data to be analyzed for bubbles and crashes.
    k : int, optional, default=10
        The number of lags used in calculating the A-statistic.
    m : int, optional, default=10
        The window size used for the first segment in calculating the S-statistic.
    n : int, optional, default=2
        The window size used for the second segment in calculating the S-statistic.
    T_star : int, optional, default=80
        The threshold time index used to determine the critical values for both A and S statistics.

    Attributes
    ----------
    critical_values : dict
        Contains the critical values for the A and S statistics:
        - 'A_critical_value': Critical value for detecting bubbles using the A-statistic.
        - 'S_critical_value': Critical value for detecting crashes using the S-statistic.
    bubbles : pandas.DataFrame
        DataFrame listing the origin and collapse dates of detected bubbles. Columns:
        - 'Origin': Start date of the bubble.
        - 'Collapse': End date of the bubble or 'On going' if no crash is detected.
    stats : pandas.DataFrame
        DataFrame containing the computed A and S statistics along with the original time series data.
    T_star : int
        Threshold time index used to determine the critical values.

    Methods
    -------
    plot()
        Plots the original time series, A-statistic, S-statistic, and highlights detected bubbles and crashes.

    Notes
    -----
    - The A-statistic is designed to detect periods of explosive growth, indicative of bubbles. 
      A bubble is flagged when the A-statistic exceeds its critical value.
    - The S-statistic detects crashes by measuring sharp declines following bubbles. A crash is 
      flagged when the S-statistic falls below its critical value.
    - The method iteratively identifies bubbles and their corresponding crashes, if any, and stores 
      them in the `bubbles` attribute.

    References
    ----------
    Whitehouse, D., Harvey, D. I., & Leybourne, S. J. (2022). Detecting bubbles and crashes in financial markets. 
    *Journal of Financial Econometrics*.

    Examples
    --------
    >>> import pandas as pd
    >>> y = pd.Series([...], index=pd.date_range('2000-01-01', periods=100))
    >>> whl_test = WHL(y, k=10, m=10, n=2, T_star=80)
    >>> print(whl_test.bubbles)  # View detected bubbles and crashes
    >>> whl_test.plot()  # Plot the time series, statistics, and detected events
    """
    
    def __init__(self, y, k=10, m=10, n=2, T_star=80) -> None:

        self.T_star = T_star
        df = y.rename('y').to_frame()
        df['delta'] = y.diff()
        df.index = range(1, len(y)+1)
        
        # A_statistic
        multiplier = np.arange(1,k+1)
        for e in range(k+1, len(y)+1):
            array = multiplier * df.delta.loc[e-k+1: e]
            df.loc[e, 'A_statistic'] = np.sum(array) / np.sqrt(np.sum(array ** 2))
        A_critical_value = np.max(df.A_statistic.loc[:T_star])
        df['bubble_flag'] = df.A_statistic > A_critical_value
        
        #S_statistic
        for e in range(m+n+1, len(df)+1):
            array1 = np.sum(df.delta.loc[e-m-n+1:e-n]) * np.sum(df.delta.loc[e-n+1:e])
            array2 = np.sqrt(np.sum(df.delta.loc[e-m-n+1:e-n]**2) * np.sum(df.delta.loc[e-n+1:e]**2))
            df.loc[e, 'S_statistic'] = array1/array2
        S_critical_value = np.min(df.S_statistic.loc[:T_star])
        df['crash_flag'] = df.S_statistic < S_critical_value
        self.critical_values = {'A_critical_value':A_critical_value, 'S_critical_value':S_critical_value}
        
        # Detect bubbles
        a_array = df[df.bubble_flag == True].index.values
        a_array = a_array[a_array >= T_star+k]
        s_array = df[df.crash_flag == True].index.values
        origin = []
        end = []
        
        if a_array.size == 0:
            return print('No bubble is detected')
        while a_array.size > 0:
            origin.append(y.index[a_array[0]-1])
            s_array = s_array[s_array>a_array[0]]
            if s_array.size > 0:
                end.append(y.index[s_array[0]-1])
                a_array = a_array[a_array >= s_array[0]+k]
            else:
                end.append('On going')
                break
        self.bubbles = pd.DataFrame({'Origin':origin, 'Collapse':end}, index=[f'bubble_{i+1}' for i in range(len(origin))])
        
        self.stats = df.drop(columns=['delta', 'bubble_flag', 'crash_flag'])
        self.stats.index = y.index
        
    def plot(self):
        fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,9))
        axs[0].plot(self.stats['y'], label='y')
        max = np.max(self.stats['y'])
        axs[1].plot(self.stats.loc[:,'A_statistic'], color='g', label='A_statistic')
        axs[1].plot(self.stats.loc[:,'S_statistic'], color='r', label='S_statistic')
        axs[1].axhline(self.critical_values['A_critical_value'], ls=':', color='g')
        axs[1].axhline(self.critical_values['S_critical_value'], ls=':', color='r')
        
        axs[0].axvline(self.stats.index[self.T_star - 1], ls='--')
        axs[1].axvline(self.stats.index[self.T_star - 1], ls='--')
        axs[0].text(self.stats.index[self.T_star], max, r'$\mathregular{T^{*}}$')
        for idx in self.bubbles.index:
            axs[0].axvline(self.bubbles.loc[idx, 'Origin'], ls='--', color='g')
            axs[1].axvline(self.bubbles.loc[idx, 'Origin'], ls='--', color='g')
            axs[0].text(self.bubbles.loc[idx, 'Origin'], max, 'Bubble detected')
            if type(self.bubbles.loc[idx, 'Collapse']) is not str:
                axs[0].axvline(self.bubbles.loc[idx, 'Collapse'], ls='--', color='r')
                axs[1].axvline(self.bubbles.loc[idx, 'Collapse'], ls='--', color='r')
                axs[0].text(self.bubbles.loc[idx, 'Collapse'], max, 'Crash detected')