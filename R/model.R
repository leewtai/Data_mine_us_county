library(randomForest)
library(glue)

svi <- read.csv('../SVI_2020_US_county.csv')
clusters <- read.csv('cluster_out.csv')

head(svi)
unwanted_vars <- c("ST", "STATE", "ST_ABBR", "STCNTY", "COUNTY", "FIPS")
unwanted <- names(svi) %in% unwanted_vars
svi <- svi[, !unwanted]

mdf <- merge(clusters[, c("cluster", "NAME")], svi, by.x='NAME', by.y='LOCATION')
dim(mdf)
dim(clusters)

dim(svi)
# right join shows that some counties are just missing, not a typo problem
mdf[is.na(mdf$cluster), c("NAME")]
svi$LOCATION[grepl('Fairfield', svi$LOCATION)]
clusters$NAME[grepl('Fairfield', clusters$NAME)]
mdf$cluster <- as.factor(mdf$cluster)
mdf[['FAKE1']] <- sample(mdf$E_HH)
mdf[['FAKE4']] <- sample(mdf$E_HBURD)
mdf[['FAKE2']] <- sample(mdf$E_AGE65)
mdf[['FAKE3']] <- sample(mdf$E_AGE17)



##The original implementation of CV and hyperparameter tunning is inefficient.
## This approach uses a more efficient implementation of cross-validation and parameter tuning.
library(caret)
library(ranger)
library(dplyr)
levels(mdf$cluster) <- make.names(levels(mdf$cluster))
# List of num.tree values to try
tree_values <- c(500, 1000, 2000, 5000)

# Create CV control
set.seed(1)
fitControl <- trainControl(
  method = "cv",
  number = 6,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

# Store results
model_results <- data.frame()
all_models <- list()

for (ntree in tree_values) {
  cat("Training model with num.tree =", ntree, "\n")
  
  model <- train(
    cluster ~ ., 
    data = mdf,
    method = "ranger",
    trControl = fitControl,
    tuneLength = 1,             
    num.tree = ntree,
    metric = "Accuracy",
    verbose = FALSE
  )
  
  all_models[[as.character(ntree)]] <- model
  
  result_row <- model$results %>%
    mutate(num.tree = ntree)
  
  model_results <- bind_rows(model_results, result_row)
}

print(model_results)

# Identify best num.tree
best_model <- model_results %>%
  arrange(desc(Accuracy)) %>%
  slice(1)

best_num_tree <- best_model$num.tree
print(paste("Best num.tree is", best_num_tree))

# Train final randomForest model using best num.tree
library(randomForest)
mod_forest <- randomForest(
  cluster ~ ., 
  data = mdf,
  ntree = best_num_tree,
  importance = TRUE
)

# Plot and save variable importance
png("forest_imp.png")
varImpPlot(mod_forest, type = 2)
dev.off()