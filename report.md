#summary

This project is trying to see the relationship of social vulnerability index (SVI) and household structures (single or married) in different counties of the United States. To do so, ingest_census.py acquires the B11002 data via an API call from the Census website. Then, to acquire geographical information, the geo shapefiles were used. In the ts_feat_gen.py, features like slope and trends of household changes in each county. In the cluster.R file, K-means clustering and t-SNE was performed to find household trends. After clustering, random tree model was used to predict the household changes as a measurement of how closely correlated SVI and household structure is. 

#technical issues

The first issue with the code is the confusion of the variables. In the report the specific variable for analysis is B11003, but in the ingest_census.py, the API was acquiring data for B11002. 
For a project that is trying to compare differences of SVI in different household structures, B11002 would be the most appropriate as it gives insight on household composition (single, married, etc.).

There are minute issues in the code such as having a set subset of data in a function but then applying the function in a for-loop. This may not achieve the intended effect. The changes were made accordingly. 

Some of the code is through "hard" indexing such as using ilocs to locate specific columns. It works, but in the case that this is applied to more data or a different dataset, or that the sequence cannot be ensured. It would be clearer. 



#non-technical issues

It would be better if there could be more analysis or explanation of the statistic methods used in this report. For instance, as counties that did not reach the 0.6 threshold were filtered out, it may be helpful to also discuss how they look differently from the ones past 0.6. 

Overall, include titles and clearer labels instead of using the variable names so the graphs are more intuitive. It is not clear at the first glance what each graph represents and the units mean. 

Writing wise, it would have been helpful to state out what the research question is so that the conclusion and the process as a whole is focused to answering the question. It feels like the conclusion did not answer or provide insights to the question proposed initially (does the shift of household structure affect SVIs in different counties).

In displaying the different clusters, it would reveal more patterns about each clusters by displaying the average trend or multiple counties.

