
#HW 5 - LSE2117

##1) what is the project doing?

My understanding of what this project is doing is that it is looking at how married vs unmarried households have changed over time and if those changes are related to social vulnerability index metrics. To do this, it first makes a dataset by combining census data and data from the social vulnerability index dataset. Then, the ML approach the project takes is to use polynomial regression to measure how the households have changed over time, and then use k means to cluster the counties in order to find patterns between them. Finally, it uses a random forest model to determine if it is possible to use these SVI features of interest to predict household change patterns. 

##2) non-technical improvement 

I would suggest reformatting pretty much all of the visual aspects of the report. The first visual, while useful insofar as it is densely populated with important information about the nature of the factors being considered by the SVI, suffers from the odd method of segmentation and the sideways labeling makes it displeasing to the eye and harder to interpet. Further down in the report, the majority of the graphs do not have well labeled axises, which makes them difficult or sometimes impossible to read (for exmample, one can infer what "sdf$year" means, although it is not aesthetic; however, a nontechnical audience would likely have no hope of decyphering an axis like "km_bag[,3]"). Additionally, there are various points throughout the project that have completely broken visuals, such as the broken link to the random forest graph and some non-formatted latex (not human readable) at the bottom of the report. Overall, **my big suggestion** would be to go back through each and every visualization and a) make sure they are visualizing (at all), and b) if they are, make sure they would be accessible and easy to read even to a non-technical audience

##3) technical improvement 

I suggested a change to the fit_best_polynomial function, because i noticed it was constructing polynomial features manually using NumPy. Even though this ends up working out in this case, I suggest to use the built in PolynomialFeatures class from sklearn.preprocessing because it is specifically designed for this task. By making this change, the function now has better error handling, modularity, efficiency, and readability. In sum, there is a lower change of bugs in during runtime and more consistent behavior for generating polynomial regressors. I also added this changed within a try / except block structure so as to maximize error handling capability and make sure this function does not fail silently and cause confusing bugs later down the line. This way we make sure that it is capable of handling NaNs and raises an error message if the processing step ends up emptying everything out. <-- would save time on debugging and improves the applicability of the function since in the real world datasets are often messy and ill-maintained.

##4) implementation 

see suggested change in the PR, but also for a clear comparison, see below: 


### Original Version

```python
def fit_best_polynomial(X, Y, k=1):
    X_powers = X.copy()
    for i in range(2, k+1):
        X_powers = np.concatenate([X_powers, np.power(X, i)], axis=1)
    assert X_powers.shape[1] == k
    mod = LinearRegression().fit(X_powers, Y)
    return np.concatenate([mod.intercept_,
                           mod.coef_[0],
                           np.array([mod.score(X_powers, Y)])]) # R^2

```
### Suggested version 
```python
from sklearn.preprocessing import PolynomialFeatures

def fit_best_polynomial(X, Y, k=1):
    try:
        # Remove rows with NaNs in either X or Y
        valid = ~np.isnan(X).flatten() & ~np.isnan(Y).flatten()
        X = X[valid]
        Y = Y[valid]

        if len(X) == 0:
            raise ValueError("No valid data points after filtering NaNs.")

        poly = PolynomialFeatures(degree=k)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        mod = LinearRegression().fit(X_poly, Y)
        return np.concatenate([mod.intercept_, mod.coef_.flatten()[1:], [mod.score(X_poly, Y)]])
    
    except Exception as e:
        print(f"Polynomial fitting failed: {e}")
        return np.full((k + 1,), np.nan)  # intercept, k coefs, and R^2
