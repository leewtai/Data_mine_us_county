# Clusting the US Counties Household Counts

This is homework 5 for Applied Machine Learning (Spring 2025) at Columbia University.

This homework requires you to:

- Make a single pull request (PR) against this repository as your submission, this PR should satisfy the following:
  - This PR should be created under a **new branch**, named after your UNI (i.e. do not push against the main branch). See some [git basics](https://leewtai.github.io/setup/git_for_beginniners) here.
  - You need to create a new file: `report.md`, `report.ipynb`, OR `report.Rmd` that contains all of the answers to the questions listed below.
  - You will need to make **at least one change** in the existing code either under the `python` or `R` folder. This can be about the logic, the efficiency, or the readability. This should be accompanied with a comment that explains your change. More changes are encouraged but remember that the public can see the changes you've recommended.
  - On Gradescope, please just enter the link to your github pull request.


## Questions your report should answer:

- Give a summary of what you think the following project is doing. Limit your answer to one paragraph.
- The student is given a passing grade and is very upset.
  - Please suggest at least one major non-technical improvement/correction they should do (e.g. writing, graphs, etc)
  - Please suggest at least one major technical improvement/correction they should do.
- Please implement your technical recommendation above in your report. The household data is on CourseWorks `Files/census/`.

## SUMMARY

In this report, the author looks into how changes in household composition across U.S. counties relate to the Social Vulnerability Index (SVI) utilizing data from the American Community Survey collected in 2020. The features from household counts (for married and unmarried households) that are used in the model are derived from best-fit polynomial curves, and these features are then used to group counties via k-means clustering. Based on visualizations of specific counties within each cluster, a preditive random forest model is used to predict cluster membership. Overall prediction accuracy is slightly above 50%, determining  demographic and economic variables—such as poverty rates, age distributions, and housing cost burden—as being the most influential in distinguishing between household clusters.

## NON-TECHNICAL IMPROVEMENT

One improvement that this report can make is in justifying its choice of features. The student lists five features as contributing to the model, but doesn't indicate how they arrived at that point. Is there some amount of exploratory data analysis that would highlight how variations in some of those features could indicate their usefulness in modeling? Did they systematically select features using stringent feature selection methods like LASSO or RIDGE? What are the original pool of features from the data that could've been chosen?

## TECHNICAL IMPROVEMENT

For the technical improvement, I added code to the model.R file so that confusion matrices are output so that we can evaluate the performance metrics across the clusters. A major critique of this report occurs in the fact that while the prediction error is better than random guess of 25% given the four clusters, a close to a 50-50 shot could be much improved, but the reporter does not take the time to examine if there are systemic recurrences in the errors the model makes. Are the same households or clusters not getting classified correctly? Or are the rates of correct classification the same for all clusters? Thus, I inserted confusion matrices so that we can further examine where the errors in the model lie. 


## Fake Report

In this report, we explore the relationships between the [social vulnerability index](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html) and changes in the households in different counties in the US.

The social vulnerability index contains metrics on socioeconomic status, household characteristics, racial & ethnic minority status, and housing type & transportation. We will use the SVIs from the year 2020 as our input.

<image src='svi_overview.png'>

The changes in households are inferred from the American Community Survey (5 year), specifically `B11003`. We focus only on the counts for households led by married couples and unmarried singles (2 separate counts). One could imagine the trade-off between these households as an indication of changing social structure in the county. We show the data for San Francisco as an example.

<image src='R/sf_eg.png'>

Features for both household counts:

- The recent change in household counts will be inferred by fitting a best fit curve to the household counts then calculating the implied slope.
- The acceleration of household counts can be similarly derived as above but using the 2nd derivative.
- An indicator whether the implied slope **never** changed signs since 2009 to 2023.
- Actual the household count in 2022
- Whether the best fit curve **cannot** be found, this is 1 if we cannot find a best fit and -1 otherwise. 

The best fit curve is calculated by regressing the counts over time up to a cubic polynomial then the smallest degree polynomial with $$R^2>0.6$$ will be called the best fit curve (since higher polynomials always produce a higher $$R^2$$ value). Cases that never achieve the $$0.6$$ threshold will be considered too noisy for us to infer a reasonable slope or acceleration for those counties.

Here are some examples of household counts that did not reach the 0.6 threshold.

<image src='R/look_at_bad_fits.png'>

We cluster the counties based on the normalized features listed above (5 features per count so 10 features total) with the exception for the best fit curve (remains -1 and 1). We evaluate the slope and acceleration at 2022.

By using k-means clustering, we see 5 clusters as being optimal (see below). But this generated a cluster with a single county, Los Angeles, CA. This is clearly an outlier situation so we removed it and fit a 4 cluster k-means (see `R/cluster.R`).

<image src='R/kmeans_btwss_by_k_with_LA.png'>

To understand the clusters, we look at few counties in these clusters.

Cluster 1
- <image src='R/eg_cluster1_5_curve.png'>

Cluster 2
- <image src='R/eg_cluster2_5_curve.png'>

Cluster 3 - 
- <image src='R/eg_cluster3_5_curve.png'>

Cluster 4 - curves that don't match
- <image src='R/eg_cluster4_5_curve.png'>

Then we try to predict these cluster classes using random forest using the SVI (see `R/model.R` for details). The cross validation shows that the prediction error is around 52%, a much higher value than random guessing.

We vary the number of trees as a hyperparameter to see if we can surpass the default prediction error. The higher number of trees only improve the prediction by a limited amount (to 54%).

Given our model is predictive, looking at the features that were important could give us insights into important features. Over the different cross validations, the consistent features that pop up are
- household counts themselves
- the number of people over age 65
- the number of people below age 18
- the % in poverty
- the number of households where the housing cost is more than 30% of their income 

The first 2 makes sense given we were dealing with household counts. Having people below 18 would likely point to married households.

<image src='R/forest_imp.png'>

Our results over the different cross validation folds are quite robust.

#### Conclusion

Overall, we learned that factors related to poverty, age distribution, and cost of living seem to be correlated with the change in households led by married and unmarried people.

None of these are particularly surprising in hindsight. It would be more enlightening to know if the relative SVI to a county's neighboring counties would be telling of its household counts over time.

{% include lib/mathjax.html %}
