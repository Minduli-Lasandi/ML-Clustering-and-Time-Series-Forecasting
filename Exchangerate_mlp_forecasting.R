#Loadng required libraries
library(readxl)
library(neuralnet)
library(ggplot2)

# Load the Excel file with data
data <- read_excel("ExchangeUSD_Dataset.xlsx")

# Extract the 3rd column from dataset
extracted_data <- data[[3]]

#Extract the data for training and testing
train_data <- extracted_data[1:400]
test_data <- extracted_data[401:500]
print(test_data)




# Define a function to create time-delayed input vectors
create_input_vectors <- function(data, delay) {
  inputs <- matrix(NA, nrow = length(data) - delay, ncol = delay)
  for (i in 1:delay) {
    inputs[, i] <- data[i:(length(data) - delay + i - 1)]
  }
  inputs
}

# Set the maximum delay (up to t-4 level)
max_delay <- 4

# Create input vectors for training and testing
train_inputs <- create_input_vectors(train_data, max_delay)
test_inputs <- create_input_vectors(test_data, max_delay)


# Create output vectors for training and testing 
train_outputs <- train_data[(max_delay + 1):length(train_data)]
test_outputs <- test_data[(max_delay + 1):length(test_data)]



# FIX: Scale test inputs and outputs using training set parameters.
# The test set must be normalised with the same mean/SD as the training set
# so both sets live on the same scale. Scaling each set independently
# would place them on different scales and invalidate the evaluation.
train_input_mean <- colMeans(train_inputs)
train_input_sd   <- apply(train_inputs, 2, sd)
scaled_train_inputs <- scale(train_inputs)
scaled_test_inputs  <- scale(test_inputs,
                             center = train_input_mean,
                             scale  = train_input_sd)

train_output_mean <- mean(train_outputs)
train_output_sd   <- sd(train_outputs)
scaled_train_outputs <- (train_outputs - train_output_mean) / train_output_sd
scaled_test_outputs  <- (test_outputs  - train_output_mean) / train_output_sd



# Print scaled input and output matrices
for (t in 1:max_delay) {
  cat("Time Delay t-", t, ":\n")
  cat("Scaled Training Input Matrix:\n")
  print(scaled_train_inputs[, 1:t])
  cat("Scaled Training Output Vector:\n")
  print(scaled_train_outputs)
  cat("Scaled Testing Input Matrix:\n")
  print(scaled_test_inputs[, 1:t])
  cat("Scaled Testing Output Vector:\n")
  print(scaled_test_outputs)
}


# Build a named dataframe for neuralnet using column names in the formula.
# neuralnet() requires a proper formula with named columns; passing raw
# vectors is fragile and can produce unexpected behaviour.
train_df <- as.data.frame(scaled_train_inputs)
colnames(train_df) <- paste0("x", 1:max_delay)
train_df$y <- scaled_train_outputs


#------------------------  NN Models   ------------------------


# Define MLP models with different configurations
# FIX: linear.output = TRUE is used throughout because exchange rate
# prediction is a regression task. Using FALSE applies a sigmoid to the
# output layer, squashing all predictions into (0,1) which is wrong for
# continuous-valued regression.
models <- list(
  # Model number 1 - 2 hidden layers (10, 5), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(10, 5),
      linear.output = TRUE
    )
  ),
  # Model number 2 - 1 hidden layer (20), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(20),
      linear.output = TRUE
    )
  ),
  # Model number 3 - 2 hidden layers (10, 5), 3 inputs (t-1 to t-3)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3,
      data = train_df,
      hidden = c(10, 5),
      linear.output = TRUE
    )
  ),
  # Model number 4 - 1 hidden layer (15), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(15),
      linear.output = TRUE
    )
  ),
  # Model number 5 - 1 hidden layer (20), 2 inputs (t-1 to t-2)
  list(
    model = neuralnet(
      y ~ x1 + x2,
      data = train_df,
      hidden = c(20),
      linear.output = TRUE
    )
  ),
  # Model number 6 - 2 hidden layers (15, 5), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(15, 5),
      linear.output = TRUE
    )
  ),
  # Model number 7 - 1 hidden layer (10), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(10),
      linear.output = TRUE
    )
  ),
  # Model number 8 - 2 hidden layers (20, 10), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(20, 10),
      linear.output = TRUE
    )
  ),
  # Model number 9 - 1 hidden layer (5), all 4 inputs (t-1 to t-4)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3 + x4,
      data = train_df,
      hidden = c(5),
      linear.output = TRUE
    )
  ),
  # Model number 10 - 2 hidden layers (7, 10), 3 inputs (t-1 to t-3)
  list(
    model = neuralnet(
      y ~ x1 + x2 + x3,
      data = train_df,
      hidden = c(7, 10),
      linear.output = TRUE
    )
  ),
  # Model number 11 - 1 hidden layer (20), 1 input (t-1 only), tanh activation
  list(
    model = neuralnet(
      y ~ x1,
      data = train_df,
      hidden = c(20),
      act.fct = "tanh",
      linear.output = TRUE
    )
  ),
  # Model number 12 - 2 hidden layers (10, 20), 2 inputs (t-1 to t-2)
  list(
    model = neuralnet(
      y ~ x1 + x2,
      data = train_df,
      hidden = c(10, 20),
      linear.output = TRUE
    )
  )
)




# Evaluate each model
results <- data.frame(
  
  RMSE = numeric(length(models)),
  MAE = numeric(length(models)),
  MAPE = numeric(length(models)),
  sMAPE = numeric(length(models))
)

# Build the test dataframe with the same named columns as the training frame
test_df <- as.data.frame(scaled_test_inputs)
colnames(test_df) <- paste0("x", 1:max_delay)

# Loop through each model and evaluate
for (i in 1:length(models)) {
  model <- models[[i]]$model
  
  
  # Make predictions using the model
  predictions <- predict(model, test_df)
  
  # FIX: Inverse scale using TRAINING set parameters (mean/SD of train_outputs).
  # Using test_outputs statistics would reverse a different scaling than was
  # applied, producing incorrect de-normalised values.
  inv_predictions <- predictions * train_output_sd + train_output_mean
  inv_actuals     <- test_outputs   # already on original scale
  
  # Calculate evaluation metrics
  RMSE  <- sqrt(mean((inv_predictions - inv_actuals)^2))
  MAE   <- mean(abs(inv_predictions - inv_actuals))
  MAPE  <- mean(abs((inv_predictions - inv_actuals) / inv_actuals)) * 100
  sMAPE <- mean(2 * abs(inv_predictions - inv_actuals) /
                  (abs(inv_predictions) + abs(inv_actuals))) * 100
  
  # Store results in the results dataframe
  
  results[i, "RMSE"]  <- RMSE
  results[i, "MAE"]   <- MAE
  results[i, "MAPE"]  <- MAPE
  results[i, "sMAPE"] <- sMAPE
  
  # Print the results
  
  cat("Model", i, "\n")
  cat("RMSE:", RMSE, "\n")
  cat("MAE:", MAE, "\n")
  cat("MAPE:", MAPE, "%\n")
  cat("sMAPE:", sMAPE, "%\n\n")
}

# Print the final results dataframe
print(results)


#Plotting the graph for the best model 
best_model <- models[[7]]$model  

# Make predictions using the best model
predictions <- predict(best_model, test_df)

# FIX: Inverse scale using training set parameters, consistent with the
# normalisation applied above.
inv_scaled_predictions <- predictions * train_output_sd + train_output_mean
inv_scaled_actuals     <- test_outputs   # already on original scale

# Create a dataframe for plotting
plot_data <- data.frame(Desired = inv_scaled_actuals, Predicted = inv_scaled_predictions)


# Plot the predicted output vs. desired output
ggplot(plot_data, aes(x = Desired, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Desired Output", y = "Predicted Output", title = "MLP Network: Predicted vs. Desired Output") +
  theme_minimal()