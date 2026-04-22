library(readxl) 
library(NbClust)
library(factoextra)
library(cluster)
library(fpc)

#Load the excel file with the data
read_data <- read_excel("Whitewine_Dataset.xlsx") 

#Read the first 11 attributes of the dataset
data <- read_data[, 1:11]
data <- data[, sapply(data, is.numeric)]

# Scale the data
scaled_data <- scale(data)  
print(scaled_data) 

#Print the boxplot before the outlier removal
boxplot(scaled_data, outline = TRUE)


#--------------------   Removing outliers  --------------------


# Select the best threshold for z-score method
z_scores <- apply(scaled_data, 1, function(x) max(abs(x)))
for (threshold in c(2, 2.5, 3, 3.5, 4)) {
  outliers <- which(z_scores > threshold)
  print(paste("Number of outliers detected using z-score method with threshold",
              threshold, ":", length(outliers)))
}

# Identify outliers using z-score method
outliers <- which(z_scores > 3)
print(paste("Number of outliers detected using z-score method:", length(outliers)))


# Remove outliers from the data
filtered_data <- scaled_data[-outliers, ]
cat("Number of samples after outlier removal:", nrow(filtered_data))


#Print the boxplot after the outlier removal
boxplot(filtered_data, outline = TRUE) 


#--------------------   Determining the number of clusters   --------------------


# performing nb clust method
par(mar = c(2, 2, 1, 1))    # Set the margins for the graph 
nb_result <- NbClust(filtered_data , distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")


# Performing Elbow method

fviz_nbclust(filtered_data,kmeans,method = "wss")+ labs (subtitle = "Elbow method")

# Performing gap stat method
gap_stat <- clusGap(filtered_data , FUN = kmeans, nstart = 10, K.max = 10, B = 50)
plot(gap_stat, main = "Gap Statistic")


# Performing silhouette method 
set.seed(100) 
silhouette_info <- fviz_nbclust(filtered_data , FUNcluster = kmeans, method = "silhouette")
print(silhouette_info)

#--------------------  K means Clustering    --------------------


#Perform clustering 
set.seed(100)  
kmeans_model <- kmeans(filtered_data, centers = 2)
print(kmeans_model)

#Calculating the BSS/TSS ratio
BSS <- kmeans_model$betweenss
WSS <- kmeans_model$tot.withinss
TSS <- BSS + WSS
BSS_TSS_ratio <- BSS / TSS
cat("Between-Cluster Sum of Squares (BSS):", BSS, "\n")
cat("Within-Cluster Sum of Squares (WSS):", WSS, "\n")
cat("BSS/TSS Ratio:", BSS_TSS_ratio, "\n")


# Calculate sihouette score 
sil_scores <- silhouette(kmeans_model$cluster, dist(filtered_data))

fviz_silhouette(sil_scores)

avg_sil_width <- mean(sil_scores[, "sil_width"])
cat ("Average sihouette width: ",avg_sil_width)


# ----------------------  Performing PCA   ---------------------- 


# Calculate covariance matrix of the filtered data 
cov_matrix <- cov(filtered_data)

# Calculate eigenvalues and eigenvectors
eigen_data <- eigen(cov_matrix)


# Calculate cumulative score per PC
variance <- eigen_data$values / sum(eigen_data$values)
cumulative_score <- cumsum(variance) * 100
print(cumulative_score)


# Identify PCs with cumulative score > 85%
selected_pcs <- which(cumulative_score > 85)[1]
cat("Number of PC's neede:",selected_pcs)

# FIX: prcomp() does not accept n.comp; all PCs are computed and
# the desired number is selected by subsetting the scores matrix afterward
pca_result <- prcomp(filtered_data, scale = TRUE)
summary(pca_result)

transformed_data <- pca_result$x[, 1:selected_pcs]
print(transformed_data)


#Performing nb clust for new dataset 
par(mar = c(2, 2, 1, 1)) 
new_nb_result <- NbClust(transformed_data , distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
summary(new_nb_result)


# Performing elbow method for new datset 
fviz_nbclust(transformed_data,kmeans,method = "wss")+ labs (subtitle = "Elbow method")


# Performing gap stat method for new dataset 
new_gap_stat <- clusGap(transformed_data , FUN = kmeans, nstart = 10, K.max = 10, B = 50)
plot(new_gap_stat, main = "Gap Statistic")


# Performing silhouette method for new dataset
set.seed(100) 
new_silhouette_info <- fviz_nbclust(transformed_data , FUNcluster = kmeans, method = "silhouette")
print(new_silhouette_info)


#Perform clustering for new dataset 
set.seed(100)  
new_kmeans_model <- kmeans(transformed_data, centers = 2)
print(new_kmeans_model)

new_BSS <- new_kmeans_model$betweenss
new_WSS <- new_kmeans_model$tot.withinss
new_TSS <- new_BSS + new_WSS
new_BSS_TSS_ratio <- new_BSS / new_TSS
cat("Between-Cluster Sum of Squares (BSS):", new_BSS, "\n")
cat("Within-Cluster Sum of Squares (WSS):", new_WSS, "\n")
cat("BSS/TSS Ratio:", new_BSS_TSS_ratio, "\n")


# FIX: distance matrix must be computed on transformed_data (PCA scores),
# not on filtered_data, so the silhouette reflects the PCA clustering space
new_sil_scores <- silhouette(new_kmeans_model$cluster, dist(transformed_data))

fviz_silhouette(new_sil_scores)

new_avg_sil_width <- mean(new_sil_scores[, "sil_width"])
cat ("Average sihouette width: ",new_avg_sil_width)



# Calculate distance matrix
# FIX: moved outside the loop below since it does not change per iteration
dist_matrix <- dist(transformed_data)

# Calculate Calinski-Harabasz Index
new_ch_index <- cluster.stats(dist_matrix, new_kmeans_model$cluster)$ch

cat("Calinski-Harabasz Index:", new_ch_index, "\n")


# Initialize vectors to store CHI values and number of clusters
chi_values <- numeric(length = 9)  
num_clusters <- 2:10  # Range of clusters 


# Iterate through different numbers of clusters and calculate CHI values
for (i in seq_along(num_clusters)) {
  k <- num_clusters[i]
  kmeans_model <- kmeans(transformed_data, centers = k)
  chi_values[i] <- cluster.stats(dist_matrix, kmeans_model$cluster)$ch
}

# Plot CHI values against number of clusters
plot(num_clusters, chi_values, type = "b", 
     xlab = "Number of Clusters", ylab = "Calinski-Harabasz Index",
     main = "CHI Plot")