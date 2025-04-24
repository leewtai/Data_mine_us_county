library(glue)
library(Rtsne)
library(RColorBrewer)
library(cluster)
library(FactoMineR)
library(factoextra)

# Load data
df <- read.csv("../python/curve_feats_counties.csv")
df <- df[, -which(grepl('2023', names(df)))]
ts <- read.csv("../python/with_geo_household_cnt.csv")

# Impute missing values with column means before scaling
imputed_df <- df[, -1]
for (i in seq_along(imputed_df)) {
  imputed_df[is.na(imputed_df[, i]), i] <- mean(imputed_df[, i], na.rm = TRUE)
}

# Scale features
scaled_features <- scale(imputed_df)
scaled_df <- as.data.frame(scaled_features)
scaled_df[['NAME']] <- df$X

# Create binary indicators for missing slopes (pre-imputation)
scaled_df[['no_curve_married']] <- ifelse(is.na(df$married.slope_2022), 1, 0)
scaled_df[['no_curve_unmarried']] <- ifelse(is.na(df$unmarried.slope_2022), 1, 0)

# Remove 'Los Angeles County, California'
scaled_df <- scaled_df[scaled_df$NAME != 'Los Angeles County, California', ]

# Run PCA and retain enough components to explain ~90% variance
pca_result <- prcomp(scaled_df[, !(names(scaled_df) %in% c("NAME"))], center = TRUE, scale. = TRUE)
explained_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
k <- which(explained_var >= 0.9)[1]
pca_data <- pca_result$x[, 1:k]

# Run k-means clustering with multiple values of k to determine optimal number
ks <- 2:20
km_bag <- matrix(NA, ncol=3, nrow=length(ks))
for(k in ks){
  km_out <- kmeans(pca_data, centers=k, nstart=20)
  km_bag[k - 1, ] <- c(k, km_out$tot.withinss, km_out$betweenss)
}

png('kmeans_btwss_by_k3.png')
plot(km_bag[, 1], km_bag[, 3], type="b", xlab="k", ylab="Between SS")
dev.off()

# Final clustering with chosen k = 4
km_out <- kmeans(pca_data, centers=4, nstart=20)
scaled_df$cluster <- km_out$cluster

# Plot cluster centers (projected onto selected features)
cols4 <- brewer.pal(4, "Set1")
png('kmeans_4_centers_slope.png')
par(mfrow=c(2, 2))
plot(km_out$centers[, 'married.slope_2022'], km_out$centers[, 'married.slope_2022'], col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'unmarried.slope_2022'], km_out$centers[, 'unmarried.slope_2022'], col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'married.slope_2022'], km_out$centers[, 'unmarried.slope_2022'], col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
plot(km_out$centers[, 'no_curve_married'], km_out$centers[, 'no_curve_unmarried'], col=cols4, pch=16)
legend("topleft", legend=1:4, fill=cols4)
dev.off()

# t-SNE visualization
cols <- brewer.pal(4, 'Set1')
for(p in seq(5, 35, by=10)){
  tsne_results <- Rtsne(pca_data, dims = 2, perplexity = p, verbose = TRUE, max_iter = 2500)
  png(glue('perplex_{p}_tsne_iter2500.png'))
  plot(tsne_results$Y[, 1], tsne_results$Y[, 2], main=glue("Perplexity {p}"),
       col=cols[scaled_df$cluster], pch=16)
  dev.off()
}

write.csv(scaled_df, 'cluster_out.csv')

# Examine problematic curve fits
no_m_yes_u <- (scaled_df[['no_curve_married']] == 1) & (scaled_df[['no_curve_unmarried']] == 0)
yes_m_no_u <- (scaled_df[['no_curve_married']] == 0) & (scaled_df[['no_curve_unmarried']] == 1)
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
    legend("topleft", legend=c('married', 'unmarried'), fill=c('blue', 'red'))
}
dev.off()

# Visualize failed curve fits
for(vs in list(c('married.slope_2022', 'B11002_003E'), c('unmarried.slope_2022', 'B11002_012E'))){
  rep_col_var <- vs[1]
  target_var <- vs[2]
  grp = sub('\\..+', '', rep_col_var)
  rand_county <- sample(scaled_df[scaled_df[, rep_col_var] == 0, 'NAME'], 1)
  sdf <- ts[ts$NAME == rand_county, c('year', target_var)]
  png(glue('eg_{rep_col_var}_curve.png'))
  plot(sdf$year, sdf[[target_var]], main=glue("No best line for {grp} household count\n {rand_county}"))
  dev.off()
}

# Visualize time trends by cluster
for(clus in 1:4){
  rand_counties <- sample(scaled_df[scaled_df$cluster == clus, 'NAME'], 5)
  for(i in seq_along(rand_counties)){
    rc = rand_counties[i]
    sdf <- ts[ts$NAME == rc, c('year', 'B11002_003E', 'B11002_012E')]
    y1 <- (sdf[['B11002_003E']] - min(sdf[['B11002_003E']])) / (max(sdf[['B11002_003E']]) - min(sdf[['B11002_003E']]))
    y2 <- (sdf[['B11002_012E']] - min(sdf[['B11002_012E']])) / (max(sdf[['B11002_012E']]) - min(sdf[['B11002_012E']]))
    png(glue('eg_cluster{clus}_{i}_curve.png'))
    plot(sdf$year, y1, main=glue("Cluster {clus} household count\n {rc}"), col="blue", pch=16, ylim=c(0, 1))
    points(sdf$year, y2, col="red", pch=16)
    legend("topleft", fill=c("blue", "red"), legend=c("married", "unmarried"), pch=16, title='household count')
    dev.off()
  }
}


