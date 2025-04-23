Q1: summary of what the student is doing

- Student seems to be trying to see whether household count statistics explain county SVI statistics. They're trying to see if there are SVI statistics that apply to certain clusters of counties (clustered based on household count changes) and not others, and, if so, what the specific SVI statistics are.

Q2: major non technical improvement

- Clustering is based on, among other things, the household count; but the random forest prediction predicts the cluster labels based on household count in the SVI data as well; recommendation: leave household count out of the prediction algorithm

Q3: major technical improvement

- In ts_feat_gen.py, the get_best_curve function doesn't return the stats of the polynomial the has the highest R^2. It returns the stats of all polynomials fitted. This should be modified to only return the highest R^2 stat.
