from itertools import product
import csv
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# You can also obtain this via CourseWorks
df = pd.read_csv(
    'with_geo_household_cnt.csv',
    usecols=['NAME', 'state', 'county', 'INTPTLAT', 'INTPTLON',
             'B11002_003E', 'B11002_012E', 'year'])

def fit_best_polynomial(X, Y, k=1):
    """
    Fits a polynomial regression model of degree k to the input data.

    This function performs polynomial regression by transforming the input features
    to include higher-order terms up to the specified degree k. It then fits a
    linear regression model to these transformed features.

    Parameters:
    X (array-like): The input feature(s). Should be a 1D or 2D array.
    Y (array-like): The target values. Should have the same number of samples as X.
    k (int, optional): The degree of the polynomial. Defaults to 1 (linear regression).

    Returns:
    numpy.ndarray: A 1D array containing the following elements:
        - The intercept of the fitted model
        - The coefficients of the polynomial terms (in ascending order of degree)
        - The R-squared score of the fitted model

    Raises:
    ValueError: If X has more than one column (multivariate input is not supported).
    """
    X = np.atleast_2d(X) #Troubleshoot if input is 1D array
    if X.shape[0] == 1:
        X = X.T  
    X_powers = X.copy()
    for i in range(2, k+1):
        X_powers = np.concatenate([X_powers, np.power(X, i)], axis=1) #Creates array with higher-order terms
    if X.shape[1] != 1:
        raise ValueError("fit_best_polynomial only supports one input feature (univariate regression).")
    mod = LinearRegression().fit(X_powers, Y)
    intercept = np.ravel(mod.intercept_)      # Convert to 1D array
    coefficients = np.ravel(mod.coef_)        # Convert to 1D array
    r_squared = np.array([mod.score(X_powers, Y)])
    return np.concatenate([intercept, coefficients, r_squared])


def get_best_curve(sdf, census_var='B11002_003E'):
    """
    Calculates polynomial regression statistics for a given census variable.

    This function fits polynomial regression models of degrees 1, 2, and 3
    to the data and returns the combined statistics for all models.

    Parameters:
    sdf (pandas.DataFrame): A DataFrame containing 'year' and census variable columns.
    census_var (str, optional): The name of the census variable column to use.
                                Defaults to 'B11002_003E'.

    Returns:
    numpy.ndarray: A 1D array containing concatenated statistics for polynomial
                   regressions of degrees 1, 2, and 3. Each set of statistics includes
                   intercept, coefficients, and R-squared score for the respective model.
    """
    X = sdf.year.to_numpy() #Slightly changed to prevent numpy shape error
    Y = sdf[census_var].to_numpy()
    poly_stats = []
    for p in range(1, 4):
        poly_stats.append(fit_best_polynomial(X, Y, p))
    poly_stat_all = np.concatenate(poly_stats)
    return poly_stat_all


# Use smallest polynomial that passes 0.6 R^2 value
# calculate slope at 2023/2024
# calculate acceleration at 2023/2024
# calculate if there was a slope change ever
# if no best line fit exists, separate it out

def calc_slope(coefs, x):
    """
    Calculate the slope of a polynomial function at given x values.

    This function computes the slope (first derivative) of a polynomial function
    defined by the given coefficients at the specified x values.

    Parameters:
    coefs (list or array-like): Coefficients of the polynomial in ascending order of degree.
                                The first element (index 0) is assumed to be the constant term.
    x (array-like): The x values at which to calculate the slope.

    Returns:
    numpy.ndarray: An array of slope values corresponding to each input x value.

    Note:
    This function prints intermediate slope calculations for each term of the polynomial.
    """
    x = np.asarray(x, dtype=float) #Troubleshoots type errors.
    slope = np.zeros_like(x, dtype=float)
    for j in range(1, len(coefs)): 
        slope += j * coefs[j] * np.power(x, j - 1) #Changed to exclude constant term 
        
    return slope


def get_feats(row, r2_cutoff=0.6):
    r2_inds = [2, 6, 11]
    years = np.array(range(2009, 2024))
    if all(row[r2_inds] < r2_cutoff):
        return np.array([np.nan] * 5)
    for p, r2_i in enumerate(r2_inds):
        if row[r2_i] < r2_cutoff:
            continue
        coefs_start = p if p == 0 else r2_inds[p - 1] + 1
        p += 1
        coefs = row[coefs_start:r2_i]
        slope = calc_slope(coefs, years)
        # acc is slope of the slope
        acc = calc_slope([c * i for i, c in enumerate(coefs) if i > 0],
                         years)
        steady_slope = np.array([1 if all(slope > 0) or all(slope < 0) else 0])

        return np.concatenate([slope[-2:], acc[-2:], steady_slope], axis=0)


df_grp = df.groupby(['NAME', 'state', 'county'])

ts_fits = []
names = []
census_vars = ['B11002_003E', 'B11002_012E']
# grp, ind = next(iter(df_grp.groups.items()))
for grp, ind in df_grp.groups.items():
    names.append(grp[0])
    sdf = df.loc[ind, ].copy()
    is_22 = sdf.year == 2022
    best_curves = [np.concatenate([get_best_curve(sdf, cv),
                                   sdf.loc[is_22, cv].to_numpy()]) for cv in census_vars]
    curve_feats = [np.concatenate([get_feats(bc), bc[-1:]]) for bc in best_curves]
    ts_fits.append(np.concatenate(curve_feats))

ts_df = pd.DataFrame(ts_fits)
ts_df.index = names
col_names = ['slope_2022', 'slope_2023', 'acc_2022', 'acc_2023', 'steady_slope', 'val_2022']
df_col_names = [status + '-' + col for status, col in product(['married', 'unmarried'], col_names)]
ts_df.columns = df_col_names

ts_df.to_csv('curve_feats_counties.csv', quoting=csv.QUOTE_NONNUMERIC)
