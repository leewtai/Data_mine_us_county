library(randomForest)
library(glue)

svi <- read.csv('/Users/tanvi/Desktop/applied machine learning/SVI_2020_US_county.csv')
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
folds <- sample(rep(1:k, times=ceiling(nrow(mdf) / k)))
folds <- folds[1:nrow(mdf)]
hyper_param_sweep <- c(500, 1000, 2000, 5000)
rf_bag <- matrix(NA, ncol=2, nrow=k)
for(i in seq_len(k)){
  is_test <- folds == i
  param_bag <- matrix(NA, ncol=length(hyper_param_sweep), nrow=k)
  for(ki in setdiff(1:k, i)){
    is_valid <- folds == ki
    is_train <- !is_test & !is_valid
    for(j in seq_along(hyper_param_sweep)){
      mod_forest <- randomForest(cluster ~ ., data=mdf[!is_test,], ntree=best_hp)
      y_pred <- predict(mod_forest, newdata=mdf[is_test,])
      y_test <- mdf[is_test,"cluster"]
      test_names <- mdf[is_test, "NAME"]  # To track misclassified counties
      
      #Confusion matrix
      conf_mat <- table(Predicted = y_pred, Actual = y_test)
      print(glue("Fold {i}: Confusion Matrix"))
      print(conf_mat)
      
      #Per-class accuracy
      classes <- levels(y_test)
      class_acc <- rep(NA, length(classes))
      names(class_acc) <- classes
      for (cls in classes) {
        true_cls_idx <- which(y_test == cls)
        class_acc[cls] <- mean(y_pred[true_cls_idx] == cls)
      }
      print(glue("Fold {i}: Per-Class Accuracy"))
      print(round(class_acc, 3))
      
      # Store confusion matrix and per-class accuracy
      if (!exists("all_conf_mats")) all_conf_mats <- list()
      if (!exists("per_class_acc_matrix")) per_class_acc_matrix <- matrix(NA, nrow=k, ncol=length(classes))
      colnames(per_class_acc_matrix) <- classes
      all_conf_mats[[i]] <- conf_mat
      per_class_acc_matrix[i, ] <- class_acc
      
      #Track misclassified counties
      misclassified <- test_names[y_pred != y_test]
      if (!exists("misclassified_all")) misclassified_all <- list()
      misclassified_all[[i]] <- misclassified
      
      # Save final performance for the fold
      rf_bag[i, 1] <- best_hp
      rf_bag[i, 2] <- sum(diag(conf_mat)) / length(y_pred)
      
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

# Summarize
print("Average Per-Class Accuracy Across Folds:")
print(round(colMeans(per_class_acc_matrix, na.rm = TRUE), 3))

# Show counties that were often misclassified
misclassified_flat <- unlist(misclassified_all)
misclassified_freq <- sort(table(misclassified_flat), decreasing = TRUE)
head(misclassified_freq, 10)  # Top 10 frequently misclassified counties

print(rf_bag)
ntree_freq <- table(rf_bag[, 1])
print(ntree_freq)
best_hp <- names(ntree_freq)[which.max(ntree_freq)]
mod_forest <- randomForest(cluster ~ ., data=mdf,
                           ntree=best_hp, importance=TRUE)

png('forest_imp.png')
varImpPlot(mod_forest, type=2)
dev.off()
