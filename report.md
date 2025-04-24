# Clusting the US Counties Household Counts

## Summary
I believe this project is exploring the changes of houshold ocmposition accross US counties by looking at the relationship between social vulnerability index (SVI) and household counts. It focuses in on married couple and unmarried single households using survey data, fitting polynomial curves to dtect trends in these counts from 2009-2023. The project then uses k-means clustering to group counties with similar patters, ultimately finding 4 different clusters. A random forst model is then used to prict cluster membership based on SVI features. 

## Non-technical Improvement
I think the report lacks clear visualization of the relationship being explored (svi features and household counts). The student could probably create correlation heatmaps or scatterplots showing how specific features of SVI directly realte to changes in household counts. This would make the insights more meaningful. One example could be scatterplots of "% population below poverty" vs "married household slope." This would let others see the direction and strength of each relationship

## Technical Improvements
I made three changes to the model.R file. First, I removed the fake random features because I think they introduce meaningless noise that would confuse the model and potentially lead to misleading feature importance rankings. Second, I simply added a set.seed function for reproducibility (I think another student in the class did this as well). Lastly, I added stratified cross-validation folds so that each fold has a balanced representation of all the cluster classes, which would produce more stabley hyperparameter tuning and a more reliable model.