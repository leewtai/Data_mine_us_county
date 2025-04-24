Report

1) What is the project about:
Upon first read it is quite unclear what exactly the project aims to achieve, up until the conclusion. What the project seems to attempt to understand is whether there is whether the proportion of married vs unmarried households influences the social vulnerability index of the county.

3) What is a nontechnical improvement:

I think a key issue in this project is the student making bold comments which they fail to consider or properly explain. A specific example is when he produces the graph for the k-means clustering graph, he finds that the optimalnumber of clusters is 5, but is gives a cluster that is onyl Los Angeles, and claims that this is clearly an outlier. Similarly, when producing the graphs of each five example clusters, he offers no explanation to explain what we are looking at. One thing I have learnt in this class is that you really dont need to be producing state-of-the-art models, instead we must try to interpret and understand them. To make this a more actionable comment, I would say that the student could slow down, and try to focus on the written parts a bit more. Make sure that graphs and functionsare described and explained and, if not, consider whether they are adding anything at all.

This supports my comment for the first question, the project is not clear on a first read what it is even about. The thesis needs to be clearer, I understand what features we are going to be using and building our model on, but for what use?


4) What is a techincal improvement:

I changed the fit_best_polynomial function in the ts_feat_gen.py file. I adjusted the X we use for fitting our polynomial and centered it, I did this simply by subtracting the mean of X. This means that for greater polynomials, X^2 or X^3, we don't have large X terms that "blow up".