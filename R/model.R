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


k <- 6

#cleaning the code
#folds <- sample(rep(1:k, times=ceiling(nrow(mdf) / k)))
#folds <- folds[1:nrow(mdf)]
folds <- sample(rep(1:k, length.out=nrow(mdf)))

hyper_param_sweep <- c(500, 1000, 2000, 5000)
rf_bag <- matrix(NA, ncol=2, nrow=k)
for(i in seq_len(k)){
  is_test <- folds == i
  param_bag <- matrix(NA, ncol=length(hyper_param_sweep), nrow=k)
  for(ki in setdiff(1:k, i)){
    is_valid <- folds == ki
    is_train <- !is_test & !is_valid
    for(j in seq_along(hyper_param_sweep)){
        mod_forest <- randomForest(cluster ~ ., data=mdf[is_train,],
                                   ntree=hyper_param_sweep[j])
        y_pred <- predict(mod_forest, newdata=mdf[is_valid,])
        y_valid <- mdf[is_valid,"cluster"]
        conf_mat <- table(y_pred, y_valid)
        param_bag[ki, j] <- sum(diag(conf_mat)) / length(y_pred)
    }
  }
  print(glue('Cross validation result {i}'))
  print(param_bag)
  best_hp <- hyper_param_sweep[which.max(apply(param_bag, 2, mean, na.rm=TRUE))]
  print(paste('Best ntree is ', best_hp))
  mod_forest <- randomForest(cluster ~ ., data=mdf[!is_test,],
                             ntree=best_hp)
  y_pred <- predict(mod_forest, newdata=mdf[is_test,])
  y_test <- mdf[is_test,"cluster"]
  conf_mat <- table(y_pred, y_test)
  rf_bag[i, 1] <- best_hp
  rf_bag[i, 2] <- sum(diag(conf_mat)) / length(y_pred)
}

print(rf_bag)
ntree_freq <- table(rf_bag[, 1])
print(ntree_freq)
best_hp <- names(ntree_freq)[which.max(ntree_freq)]
mod_forest <- randomForest(cluster ~ ., data=mdf,
                           ntree=best_hp, importance=TRUE)

png('forest_imp.png')
varImpPlot(mod_forest, type=2)
dev.off()

