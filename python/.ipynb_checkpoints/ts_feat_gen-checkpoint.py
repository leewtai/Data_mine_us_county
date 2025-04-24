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
    X_powers = X.copy()
    for i in range(2, k+1):
        X_powers = np.concatenate([X_powers, np.power(X, i)], axis=1)
    assert X_powers.shape[1] == k
    mod = LinearRegression().fit(X_powers, Y)
    return np.concatenate([mod.intercept_,
                           mod.coef_[0],
                           np.array([mod.score(X_powers, Y)])]) # R^2


def get_best_curve(sdf, census_var='B11002_003E'):
    X = sdf.year.to_numpy().reshape(-1, 1)
    Y = sdf[census_var].to_numpy().reshape(-1, 1)
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
    # assumes getting a linear model
    slope = 0 * x
    for j, coef in enumerate(coefs):
        # first term will always be 0
        comp = j * coef * np.power(x, float(j - 1))
        slope = slope + comp
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
