library(glue)
library(Rtsne)
library(RColorBrewer)

df <- read.csv("../python/curve_feats_counties.csv")
df <- df[, -which(grepl('2023', names(df)))]
ts <- read.csv("../python/with_geo_household_cnt.csv")

scaled_df <- apply(df[, -1], 2, scale, center=TRUE, scale=TRUE)
scaled_df <- as.data.frame(scaled_df)
scaled_df[['NAME']] <- df$X
head(scaled_df)

#Revised Normalization
scaled_df <- df
#1 - Log transform raw household count variables 
scaled_df <- scaled_df %>%
  mutate(
    married_log = log1p(B11002_003E_2022), #assuming this is the variable name for raw count
    unmarried_log = log1p(B11002_012E_2022)
  )
#2 - Centering by median and scaling by IQR (compared to mean and standard deviation)
robust_cols <- c("married.slope_2022", "married.acceleration_2022",
                 "unmarried.slope_2022", "unmarried.acceleration_2022")
for (col in robust_cols) {
  med <- median(scaled_df[[col]], na.rm = TRUE)
  iqr <- IQR(scaled_df[[col]], na.rm = TRUE)
  scaled_df[[col]] <- (scaled_df[[col]] - med) /iqr
}
#3 - Leaving binary variables un-normalized

#4 - Impute any remaining NAs with 0 
scaled_df[is.na(scaled_df)] <- 0

scaled_df[['no_curve_married']] <- (is.na(scaled_df[['married.slope_2022']]) - 0.5) * 2
scaled_df[['no_curve_unmarried']] <- (is.na(scaled_df[['unmarried.slope_2022']]) - 0.5) * 2
for(i in seq_len(ncol(scaled_df))){
  is_na_vals <- is.na(scaled_df[, i])
  scaled_df[is_na_vals, i] <- 0 # replaced with mean, just so kmeans will run!
}

no_m_yes_u <- (scaled_df[['no_curve_married']] == 1) & (scaled_df[['no_curve_unmarried']] == -1)
yes_m_no_u <- (scaled_df[['no_curve_married']] == -1) & (scaled_df[['no_curve_unmarried']] == 1)
no_m_no_u <- (scaled_df[['no_curve_married']] == 1) & (scaled_df[['no_curve_unmarried']] == 1)
scenarios <- list(no_m_yes_u, yes_m_no_u, no_m_no_u)
png('look_at_bad_fits.png', 1200, 500)
par(mfrow=c(1, 3))
for(i in seq_along(scenarios)){
    s <- scenarios[[i]]
    rand_county <- sample(scaled_df[s, "NAME"], 1)
    sdf <- ts[ts$NAME == rand_county, c('year', 'B11002_003E', 'B11002_012E')]
    y_range <- c(min(sdf[, -1]), max(sdf[, -1]))
    plot(sdf$year, sdf[, 2], col="blue", pch=16, ylim=y_range)
    points(sdf$year, sdf[, 3], col="red", pch=16)
    legend("topleft", legend=c('married', 'unmarried'),
           fill=c('blue', 'red'))
}
dev.off()

scaled_df <- scaled_df[scaled_df$NAME != 'Los Angeles County, California', ]
ks <- 2:20
km_bag <- matrix(NA, ncol=3, nrow=length(ks))
for(k in ks){
  km_out <- kmeans(scaled_df[, -which(names(scaled_df) == 'NAME')], centers=k, nstart=20)
  km_bag[k - 1, ] <- c(k, km_out$tot.withinss, km_out$betweenss)
}

png('kmeans_btwss_by_k3.png')
plot(km_bag[, 1], km_bag[, 3])
dev.off()

km_bag
km_out <- kmeans(scaled_df[, -which(names(scaled_df) == 'NAME')], centers=4, nstart=20)
table(km_out$cluster) 
scaled_df[km_out$cluster == 3,] # Los Angeles is its own cluster!

cols4 <- brewer.pal(4, "Set1")
png('kmeans_4_centers_slope.png')
par(mfrow=c(2, 2))
plot(km_out$centers[, 'married.val_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'married.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'unmarried.val_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'unmarried.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'married.slope_2022'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'unmarried.slope_2022'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'no_curve_married'],
     # km_out$centers[, 'unmarried.slope_2022'])
     km_out$centers[, 'no_curve_unmarried'],
     col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
dev.off()


cols <- brewer.pal(4, 'Set1')
for(p in seq(5, 35, by=10)){
tsne_results <- Rtsne(scaled_df,
                      dims = 2, perplexity = p, verbose = TRUE, max_iter = 2500)
png(glue('perplex_{p}_tsne_iter2500.png'))
plot(tsne_results$Y[, 1], tsne_results$Y[, 2], main=glue("Perplexity {p}"),
     col=cols[km_out$cluster])
dev.off()
}

scaled_df[['cluster']] <- km_out$cluster
write.csv(scaled_df, 'cluster_out.csv')


target_vars <- list(c('married.slope_2022', 'B11002_003E'),
                    c('unmarried.slope_2022', 'B11002_012E'))
for(vs in target_vars){
  target_var <- vs[2]
  rep_col_var <- vs[1]
  grp = sub('\\..+', '', rep_col_var)
  # see what bad looks like
  rand_county <- sample(scaled_df[scaled_df[, rep_col_var] == 0, 'NAME'], 1)
  sdf <- ts[ts$NAME == rand_county, c('year', target_var)]
  png(glue('eg_{rep_col_var}_curve.png'))
  plot(sdf$year, sdf[[target_var]], main=glue("No best line for {grp} household count\n {rand_county}"))
  dev.off()
}

# Vis the clusters
for(clus in 1:4){
  # get 3 counties
  rand_counties <- sample(scaled_df[km_out$cluster == clus, 'NAME'], 5)
  for(i in seq_along(rand_counties)){
      rc = rand_counties[i]
      sdf <- ts[ts$NAME == rc, c('year', 'B11002_003E', 'B11002_012E')]
      y1 <- sdf[['B11002_003E']]
      y1 <- (y1 - min(y1)) / (max(y1) - min(y1))
      y2 <- sdf[['B11002_012E']]
      y2 <- (y2 - min(y2)) / (max(y2) - min(y2))
      png(glue('eg_cluster{clus}_{i}_curve.png'))
      plot(sdf$year, y1, main=glue("Cluster {clus} household count\n {rc}"), col="blue", pch=16, ylim=c(0, 1))
      points(sdf$year, y2, col="red", pch=16)
      legend("topleft", fill=c("blue", "red"), legend=c("married", "unmarried"), pch=16, title='household count')
      dev.off()
  }
}
# large variation in values

#SUGGESTION FOR DISPLAYING TIME-SERIES (y1 and y2) OF RANDOMLY SELECTED COUNTIES 
#IN EACH CLUSTER 
#Instead, display all 5 randomly selected counties from each cluster on one graph 
#along with cluster's average time series for y1 and y2. 
#This requires the following changes to the code above:
#1) For each cluster, all county names are pulled (for average ts calculation)
#2) Plot each cluster's average ts as thick solid lines (different colors for married
#and unmarried), and overlay 5 sampled counties as dashed lines
#3) par(mfrow=c(1,4)) such that all four cluster plots can be analyzed side by side
#WHAT DOES THIS DO?
#1) Provides user with a better understanding of the within-cluster variation 
#2) Allows for hypothesis generation regarding what clusters currently mean 
#as well as different ways of normalizing or engineering features such that 
#the clusters are more meaningful
#CODE ->
#Create different colors for married and unmarried time series
avg_cols <- c(married = "blue", unmarried = "red")
sample_cols <- c(married = "skyblue", unmarried = "pink")

#Final produced image for report
set.seed(123)
years <- sort(unique(ts$year))
png("avg_ts_and_sampled_counties.png", width = 1400, height = 500)
par(mfrow = c(1, 4), mar = c(4, 4, 2, 1))

for(clus in 1:4) {
  #How many counties are in each cluster
  members <- scaled_df$NAME[km_out$cluster == clus]
  n <- length(members)
  
  #Matrices for each county's scaled y1 and y2
  y1_mat <- matrix(NA, nrow = length(years), ncol = n)
  y2_mat <- matrix(NA, nrow = length(years), ncol = n)
  
  #Scaled ts for each county in cluster 
  for (j in seq_along(members)) {
    rc <- members[j]
    tmp <- ts[ts$NAME == rc, c("year", "B11002_003E", "B11002_012E")]
    y1 <- tmp$B11002_003E 
    y1 <- (y1 - min(y1)) / (max(y1) - min(y1))
    y2 <- tmp$B11002_012E
    y2 <- (y2 - min(y2)) / (max(y2) - min(y2))
    y1_mat[, j] <- y1
    y2_mat[, j] <- y2
  }
  
  #Average scaled ts for each cluster 
  avg_y1 <- rowMeans(y1_mat, na.rm = TRUE)
  avg_y2 <- rowMeans(y2_mat, na.rm = TRUE)
  
  #Sampling 5 counties - slightly changing sampling strat from original code because
  #good to have index of sampled counties for plotting
  samp_idx <- sample(seq_len(n), 5)
  sampled_names <- members[samp_idx]
  
  #Plot avg ts, overlay sampled counties, and include legend
  plot(years, avg_y1, type="l", lwd=2, col=avg_cols["married"],
       ylim=c(0,1), xlab="Year", ylab="Scaled count",
       main=glue("Cluster {clus}"))
  lines(years, avg_y2,      lwd=2, col=avg_cols["unmarried"])
  for (k in samp_idx) {
    lines(years, y1_mat[,k], lty=2, col=sample_cols["married"])
    lines(years, y2_mat[,k], lty=2, col=sample_cols["unmarried"])
  }
    legend("topright",
         legend=c("Avg married","Avg unmarried",
                  "Sample married","Sample unmarried"),
         col  = c(avg_cols, sample_cols),
         lty  = c(1,1,2,2),
         lwd  = c(2,2,1,1),
         bty="n")
}
dev.off()




