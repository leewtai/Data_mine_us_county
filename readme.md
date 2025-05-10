Project Summary:
This report is aiming to examine the relationship between the social vulnerability index (SVI) and changes in the households in different counties in the US, specifically the counts of married versus unmarried households, from 2009-2023. The author uses the following five features for both household counts: 1. recent change in household counts which will be inferred by fitting a best fit curve; 1. acceleration of household counts; 3. an indicator whether the implied slope never changed signs since 2009 to 2023; 4. 2022 actual household count, and 5. whether the best fit curve cannot be found. The author then uses k-means to obtain 5 clusters as optimal. However, they remove LA County as its own cluster and re-obtain 4 clusters. Finally, they predict the cluster classes using random forest and the SVI as predictors. Overall, they conclude that changes in households led by married and unmarried people are related to SVI factors such as poverty, age distribution, and cost of living.

Non-technical improvement:
When they show the example county plot for each of the clusters 1-4, this is not a good way to characterize the clusters. They also don't explain them at all or label the Y axis. It would be better to overlay a couple of counties from each cluster or find some other measures to average across the counties in each cluster to explain their differences. 

Technical improvement:
I added PCA before we do the clustering and found that 90% of the variance in counties can be attributed to 7 principal components. This reduces sparsity so clustering will work better. 

