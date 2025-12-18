library(dplyr)
library(ggplot2)
library(lubridate)
library(fixest)
library(gamlr)
library(MASS)
library(Matrix)
library(glmnet)
library(writexl)
library(stargazer)

holidays <- read.csv("holidays_events.csv")
oil <- read.csv("oil.csv")
sample_submission <- read.csv("sample_submission.csv")
stores <- read.csv("stores.csv")
test <- read.csv("test.csv")
train <- read.csv("train.csv")
transactions <- read.csv("transactions.csv")

# Data Preprocessing
train$date <- as.Date(train$date)
test$date <- as.Date(test$date)
oil$date <- as.Date(oil$date)
holidays$date <- as.Date(holidays$date)
holidays <- holidays %>% 
  rename(holiday_type = type)
train$family <- factor(train$family)

# Merge train data with store metadata, oil prices, holidays data
train <- train %>% left_join(stores, by = "store_nbr")
train <- train %>% left_join(oil, by = "date")
train <- train %>% left_join(holidays, by = "date")

# Fill missing oil prices with previous values
train$dcoilwtico<- zoo::na.locf(train$dcoilwtico, na.rm = FALSE)

#Take out the effect of earthquake happened on 2016-04-16
train <- train %>%
  filter(date <= as.Date("2016-04-15"))
train$earthquake_dummy <- ifelse(train$date >= as.Date("2016-04-16") & train$date <= as.Date("2016-05-16"), 1, 0)

##Data Description
ggplot(train, aes(x = date, y = sales)) +
  geom_line() +
  labs(title = "Sales Over Time", x = "Date", y = "Sales")

#1.Multiple Linear Regression
# Dummy for Holidays, Month, 
# Payday(Assuming 15th and end of the month as paydays, with a 3-day window), Earthquakes
train$holiday_dummy <- ifelse(!is.na(train$holiday_type), 1, 0)  # 1 if it's a holiday, else 0
train$month <- as.factor(month(train$date))
month_dummies <- model.matrix(~ month - 1, data = train)  # Creates dummies for each month
train <- cbind(train, month_dummies)  # Add them to the dataset
train$payday_dummy <- ifelse(day(train$date) %in% c(14:17, 28:31), 1, 0)
train$earthquake_dummy <- ifelse(train$date >= as.Date("2016-04-16") & train$date <= as.Date("2016-05-16"), 1, 0)

train <- train %>%
  mutate(payday_dummy_lag1 = lag(payday_dummy, 1),
         payday_dummy_lag2 = lag(payday_dummy, 2),
         payday_dummy_lag3 = lag(payday_dummy, 3),
         payday_dummy_lag4 = lag(payday_dummy, 4),
         payday_dummy_lag5 = lag(payday_dummy, 5))

#1.City and State Fixed Effects: fixed effect regression
train$city <- as.factor(train$city)
train$state <- as.factor(train$state)

fe_model <- feols(sales ~ onpromotion + dcoilwtico + holiday_dummy + 
                               payday_dummy + month 
                             | store_nbr + family + city + state, data = train)
summary(fe_model)
etable(fe_model)


##2.Lasso
## Lasso Model 1: Basic Features Model
# Set reference level for month, family, city, and state based on the most frequent level
most_frequent_month <- names(sort(table(train$month), decreasing = TRUE))[1]
train$month <- relevel(train$month, ref = most_frequent_month)

most_frequent_family <- names(sort(table(train$family), decreasing = TRUE))[1]
train$family <- relevel(train$family, ref = most_frequent_family)

most_frequent_city <- names(sort(table(train$city), decreasing = TRUE))[1]
train$city <- relevel(train$city, ref = most_frequent_city)

most_frequent_state <- names(sort(table(train$state), decreasing = TRUE))[1]
train$state <- relevel(train$state, ref = most_frequent_state)

train <- na.omit(train)  # Removes all rows with any NA values

# Step 1: Create Sparse Matrix for Predictors
X_sparse_model1 <- sparse.model.matrix(sales ~ onpromotion + dcoilwtico + holiday_dummy + 
                                         payday_dummy + payday_dummy_lag1 + payday_dummy_lag2 + 
                                         payday_dummy_lag3 + payday_dummy_lag4 + payday_dummy_lag5 +
                                         month + store_nbr + family + city + state, data = train)[,-1]

# Step 2: Define Response Variable
y_model1 <- train$sales

# Step 3: Fit the LASSO Model using gamlr() (using cross-validation lambda)
lasso_model1 <- gamlr(X_sparse_model1, y_model1, family = "gaussian", gamma = 1e-4)

# Step 4: Plot the Regularization Path
plot(lasso_model1, main = "LASSO Regularization Path - Model 1")

# Step 5: Get Selected Features (Non-zero coefficients)
selected_features_model1 <- coef(lasso_model1)[which(coef(lasso_model1) != 0), ]
print(selected_features_model1)


## Lasso Model 2: Interaction Features Model (with payday lags but no lag interactions)
X_sparse_model2 <- sparse.model.matrix(sales ~ 
                                                 onpromotion * month +  # Interaction between promotion and month
                                                 onpromotion * dcoilwtico +  # Interaction between promotion and oil price
                                                 payday_dummy * onpromotion +  # Interaction between payday and promotion
                                                 holiday_dummy * onpromotion +  # Interaction between holiday and promotion
                                                 store_nbr * family +  # Interaction between store and family
                                                 family * month +  # Interaction between family and month
                                                 city * state +  # Interaction between city and state
                                                 payday_dummy * dcoilwtico +  # Interaction between payday and oil price
                                                 payday_dummy_lag1 + payday_dummy_lag2 + payday_dummy_lag3 + 
                                                 payday_dummy_lag4 + payday_dummy_lag5 +  # Including payday lags as features
                                                 store_nbr + family + city + state, data = train)[,-1]


# Step 2: Define Response Variable
y_model2 <- train$sales

# Step 3: Fit the LASSO Model using gamlr() (using cross-validation lambda)
lasso_model2 <- gamlr(X_sparse_model2, y_model2, family = "gaussian", gamma = 1e-4)

# Step 4: Plot the Regularization Path
plot(lasso_model2, main = "LASSO Regularization Path - Model 2")

# Step 5: Get Selected Features (Non-zero coefficients)
selected_features_model2 <- coef(lasso_model2)[which(coef(lasso_model2) != 0), ]
print(selected_features_model2)


## Lasso Model 3: Interactions Model with Lags
X_sparse_model3 <- sparse.model.matrix(sales ~ 
                                         onpromotion * month +  # Interaction between promotion and month
                                         onpromotion * dcoilwtico +  # Interaction between promotion and oil price
                                         payday_dummy * onpromotion +  # Interaction between payday and promotion
                                         holiday_dummy * onpromotion +  # Interaction between holiday and promotion
                                         store_nbr * family +  # Interaction between store and family
                                         family * month +  # Interaction between family and month
                                         city * state +  # Interaction between city and state
                                         payday_dummy * dcoilwtico +  # Interaction between payday and oil price
                                         store_nbr + family + city + state +
                                         payday_dummy + payday_dummy_lag1 + payday_dummy_lag2 + 
                                         payday_dummy_lag3 + payday_dummy_lag4 + payday_dummy_lag5 +
                                         payday_dummy:payday_dummy_lag1 + payday_dummy:payday_dummy_lag2 +
                                         payday_dummy_lag1:payday_dummy_lag2 + payday_dummy:payday_dummy_lag3 +
                                         payday_dummy_lag1:payday_dummy_lag3 + payday_dummy_lag2:payday_dummy_lag3 +
                                         payday_dummy:payday_dummy_lag4 + payday_dummy_lag1:payday_dummy_lag4 +
                                         payday_dummy_lag2:payday_dummy_lag4 + payday_dummy:payday_dummy_lag5 +
                                         payday_dummy_lag1:payday_dummy_lag5 + payday_dummy_lag2:payday_dummy_lag5 +
                                         month:payday_dummy + month:payday_dummy_lag1 + 
                                         month:payday_dummy_lag2 + month:payday_dummy_lag3 +
                                         month:payday_dummy_lag4 + month:payday_dummy_lag5,
                                       data = train)[,-1]

y_model3 <- train$sales


lasso_model3 <- gamlr(X_sparse_model3, y_model3, family = "gaussian", gamma = 1e-4)

plot(lasso_model3, main = "LASSO Regularization Path - Model 3")

selected_features_model3 <- coef(lasso_model3)[which(coef(lasso_model3) != 0), ]
print(selected_features_model3)


## Lasso Model 4: Seasonality & Trend Model
train$time <- as.numeric(as.Date(train$date) - min(as.Date(train$date)))


X_sparse_model4 <- sparse.model.matrix(sales ~ 
                                         time + I(time^2) + I(time^3) +  # Capture trend with polynomial terms
                                         onpromotion + dcoilwtico + holiday_dummy + 
                                         payday_dummy + month + store_nbr + family + city + state, 
                                       data = train)[,-1]

y_model4 <- train$sales

lasso_model4 <- gamlr(X_sparse_model4, y_model4, family = "gaussian", gamma = 1e-4)

plot(lasso_model4, main = "LASSO Regularization Path - Model 4")

selected_features_model4 <- coef(lasso_model4)[which(coef(lasso_model4) != 0), ]
print(selected_features_model4)


## Elastic Net Model: 
# Create sparse model matrix for Elastic Net (Model 5)
X_sparse_model5 <- sparse.model.matrix(sales ~ onpromotion + dcoilwtico + holiday_dummy + 
                                         payday_dummy + month + store_nbr + family + city + state, 
                                       data = train)[,-1]

y_model5 <- train$sales

# Fit Elastic Net model (non-cross-validated, alpha = 0.5)
# Use a specific lambda sequence to match LASSO (e.g., log(lambda) from 2 to 6)
lambda_seq <- exp(seq(2, 6, length.out = 100))  # Match LASSO lambda range
elastic_net_model5 <- glmnet(X_sparse_model5, y_model5, alpha = 0.5, family = "gaussian", 
                             lambda = lambda_seq)

# Set larger top margin to avoid overlap
par(mar = c(5, 4, 6, 2) + 0.1)  # Increase the top margin

# Plot Elastic Net coefficient paths to match LASSO style
plot(elastic_net_model5, xvar = "lambda", 
     main = "Elastic Net Regularization Path - Model 5", 
     label = TRUE,  # Label the lines with maximum coefficients
     ylim = c(0, max(abs(coef(elastic_net_model5))))  # Adjust y-axis to fit coefficients
)



# Set larger top margin to avoid overlap
par(mar = c(5, 4, 6, 2) + 0.1)  # Increase the top margin (default is 4)
plot(elastic_net_model5, main = "Elastic Net Regularization Path - Model 5")


# Cross-Validation and Model Fitting for All Models

# 1. Cross-validation for all models (Lasso Models 1-4 and Elastic Net Model 5)
cv_model1 <- cv.gamlr(X_sparse_model1, y_model1, nfold = 10)
cv_model2 <- cv.gamlr(X_sparse_model2, y_model2, nfold = 10)
cv_model3 <- cv.gamlr(X_sparse_model3, y_model3, nfold = 10)
cv_model4 <- cv.gamlr(X_sparse_model4, y_model4, nfold = 10)
cv_model5 <- cv.glmnet(X_sparse_model5, y_model5, alpha = 0.5, nfolds = 10)

# 2. Extract optimal lambda values for all models using the 1SE rule
lambda1 <- cv_model1$lambda.1se
lambda2 <- cv_model2$lambda.1se
lambda3 <- cv_model3$lambda.1se
lambda4 <- cv_model4$lambda.1se
lambda5 <- cv_model5$lambda.1se  

# 3. Find AIC at the optimal lambda position for each model

# AIC Calculation for Lasso Models 1-4
aic_pos1 <- which.min(abs(lasso_model1$lambda - lambda1))
aic_pos2 <- which.min(abs(lasso_model2$lambda - lambda2))
aic_pos3 <- which.min(abs(lasso_model3$lambda - lambda3))
aic_pos4 <- which.min(abs(lasso_model4$lambda - lambda4))

# Extract AIC values at optimal lambda positions for Lasso models
AIC1 <- AIC(lasso_model1)[aic_pos1]
AIC2 <- AIC(lasso_model2)[aic_pos2]
AIC3 <- AIC(lasso_model3)[aic_pos3]
AIC4 <- AIC(lasso_model4)[aic_pos4]

# 4. AIC Calculation for Elastic Net Model 5 (manual calculation using residual sum of squares)
coef_model5 <- coef(cv_model5, s = "lambda.1se")
non_zero_coefs5 <- sum(coef_model5 != 0)

# Predict using the best lambda (lambda.1se) for Elastic Net model 5
predictions5 <- predict(cv_model5, newx = X_sparse_model5, s = "lambda.1se")
residual_sum_squares5 <- sum((y_model5 - predictions5)^2)
n5 <- length(y_model5)

# Calculate AIC manually: AIC = n * log(RSS/n) + 2k, where k is the number of non-zero coefficients
AIC5 <- n5 * log(residual_sum_squares5 / n5) + 2 * non_zero_coefs5

# 5. Create a vector with all AIC values for comparison
AIC_values <- c(AIC1, AIC2, AIC3, AIC4, AIC5)
names(AIC_values) <- c("Model 1", "Model 2", "Model 3", "Model 4", "Elastic Net")

# Print AIC values for all models
print(AIC_values)

# 6. Select the best model based on the lowest AIC value
best_model <- names(which.min(AIC_values))
cat("Best model based on AIC:", best_model, "\n")

## Lambda Plots
# 1. Plot Lasso Model 1 (cv.gamlr output for Lasso model)
par(mfrow = c(2, 3), mar = c(4, 4, 2, 2))
par(mar = c(5, 4, 6, 2) + 0.1)  # Adjust margins for better spacing
plot(cv_model1, main = "Lasso Regularization Path - Model 1", xvar = "lambda", label = TRUE)

# 2. Plot Lasso Model 2 (cv.gamlr output for Lasso model)
plot(cv_model2, main = "Lasso Regularization Path - Model 2", xvar = "lambda", label = TRUE)

# 3. Plot Lasso Model 3 (cv.gamlr output for Lasso model)
plot(cv_model3, main = "Lasso Regularization Path - Model 3", xvar = "lambda", label = TRUE)

# 4. Plot Lasso Model 4 (cv.gamlr output for Lasso model)
plot(cv_model4, main = "Lasso Regularization Path - Model 4", xvar = "lambda", label = TRUE)

# 5. Plot Elastic Net Model 5 (cv.glmnet output for Elastic Net model)
par(mar = c(5, 4, 6, 2) + 0.1)  # Adjust margins for better spacing
plot(cv_model5, main = "Elastic Net Regularization Path - Model 5", xvar = "lambda", label = TRUE)


#As Model 3 is selected, rerun it on the entire dataset
# Step 1: Create Sparse Matrix for Predictors (no splitting, using entire dataset)
X_sparse_model3_full <- sparse.model.matrix(sales ~ 
                                              onpromotion * month +  # Interaction between promotion and month
                                              onpromotion * dcoilwtico +  # Interaction between promotion and oil price
                                              payday_dummy * onpromotion +  # Interaction between payday and promotion
                                              holiday_dummy * onpromotion +  # Interaction between holiday and promotion
                                              store_nbr * family +  # Interaction between store and family
                                              family * month +  # Interaction between family and month
                                              city * state +  # Interaction between city and state
                                              payday_dummy * dcoilwtico +  # Interaction between payday and oil price
                                              store_nbr + family + city + state +
                                              payday_dummy + payday_dummy_lag1 + payday_dummy_lag2 + 
                                              payday_dummy_lag3 + payday_dummy_lag4 + payday_dummy_lag5 +
                                              payday_dummy:payday_dummy_lag1 + payday_dummy:payday_dummy_lag2 +
                                              payday_dummy_lag1:payday_dummy_lag2 + payday_dummy:payday_dummy_lag3 +
                                              payday_dummy_lag1:payday_dummy_lag3 + payday_dummy_lag2:payday_dummy_lag3 +
                                              payday_dummy:payday_dummy_lag4 + payday_dummy_lag1:payday_dummy_lag4 +
                                              payday_dummy_lag2:payday_dummy_lag4 + payday_dummy:payday_dummy_lag5 +
                                              payday_dummy_lag1:payday_dummy_lag5 + payday_dummy_lag2:payday_dummy_lag5 +
                                              month:payday_dummy + month:payday_dummy_lag1 + 
                                              month:payday_dummy_lag2 + month:payday_dummy_lag3 +
                                              month:payday_dummy_lag4 + month:payday_dummy_lag5,
                                            data = train)[,-1]

# Step 2: Define Response Variable (no splitting, using entire dataset)
y_model3_full <- train$sales

# Step 3: Fit the LASSO Model using gamlr() (using entire dataset without splitting)
lasso_model3_full <- gamlr(X_sparse_model3_full, y_model3_full, family = "gaussian", gamma = 1e-4)

# Step 4: Plot the Regularization Path for full dataset
plot(lasso_model3_full, main = "LASSO Regularization Path - Model 3 (Full Dataset)")

# Step 5: Get Selected Features (Non-zero coefficients) for full dataset
selected_features_model3_full <- coef(lasso_model3_full)[which(coef(lasso_model3_full) != 0), ]
print(selected_features_model3_full)

# Convert to a data frame for better presentation
features_df_full <- data.frame(Feature = names(selected_features_model3_full), 
                               Coefficient = as.numeric(selected_features_model3_full))

# Write to Excel
writexl::write_xlsx(features_df_full, "selected_model3.xlsx")



#Compute Deviance
selected_lambda <- lasso_model3_full$lambda[which.min(AIC(lasso_model3_full))]
# Or use the one-standard-error rule
selected_lambda_1se <- lasso_model3_full$lambda[which.min(AIC(lasso_model3_full)) + 1]

# Get predictions at the selected lambda
predictions <- predict(lasso_model3_full, X_sparse_model3_full, lambda=selected_lambda, type="response")

# Calculate null deviance (deviance of the null model with just the intercept)
null_deviance <- sum((y_model3_full - mean(y_model3_full))^2)

# Calculate residual deviance (deviance of your fitted model)
residual_deviance <- sum((y_model3_full - predictions)^2)

# Calculate percent deviance explained
deviance_explained <- (null_deviance - residual_deviance) / null_deviance * 100

cat("Null Deviance:", null_deviance, "\n")
cat("Residual Deviance:", residual_deviance, "\n")
cat("Deviance Explained:", deviance_explained, "%\n")



#3.Time Series Analysis
# Aggregate train data by date, using existing holiday_dummy, payday_dummy, and month
train_agg <- train %>% 
  group_by(date) %>% 
  summarise(
    total_sales = sum(sales, na.rm = TRUE),
    avg_onpromotion = mean(onpromotion, na.rm = TRUE),
    avg_dcoilwtico = mean(dcoilwtico, na.rm = TRUE),
    holiday_dummy = ifelse(any(!is.na(holiday_dummy)), max(holiday_dummy, na.rm = TRUE), 0),  # Handle NA explicitly
    payday_dummy = max(payday_dummy, na.rm = TRUE),
    month = first(month)
  ) %>% 
  ungroup()

# Add time index
train_agg <- train_agg %>% 
  mutate(
    year = year(date),
    time = (year - min(year)) * 12 + as.numeric(month(date)),  # Numeric time index
    log_sales = log(total_sales + 1),
    lag_log_sales = lag(log_sales, n = 1, default = NA)  # Lagged log sales
  )

# Remove the first row (NA for lag)
train_agg <- train_agg[-1, ]

train_agg <- train_agg %>% 
  mutate(
    lag_payday_dummy_1 = lag(payday_dummy, 1, default = 0),
    lag_payday_dummy_2 = lag(payday_dummy, 2, default = 0),
    lag_payday_dummy_3 = lag(payday_dummy, 3, default = 0)
  )

# Create additional lag 7 variable
train_agg <- train_agg %>% 
  mutate(
    lag_log_sales_1 = lag(log_sales, 1, default = NA),  # AR(1)
    lag_log_sales_2 = lag(log_sales, 2, default = NA),  # AR(2)
    lag_log_sales_7 = lag(log_sales, 7, default = NA)   # Lag 7 for seasonal pattern
  )

# Remove rows with NAs for the lags
train_agg_with_lags <- train_agg %>% filter(!is.na(lag_log_sales_7))

# Fit model with AR terms and lag 7
sales_model <- glm(
  log_sales ~ lag_log_sales_1 + lag_log_sales_2 + lag_log_sales_7 +
    time + month + avg_onpromotion + avg_dcoilwtico + 
    payday_dummy + lag_payday_dummy_1 + lag_payday_dummy_2 + lag_payday_dummy_3, 
  data = train_agg_with_lags
)

summary(sales_model)
stargazer(sales_model, type = "text")

# Plot actual vs. fitted log sales
plot(
  train_agg_with_lags$time, 
  train_agg_with_lags$log_sales, 
  type = "l", 
  col = "green", 
  xlab = "Time (Months since Start)", 
  ylab = "Log Total Sales", 
  main = "Log Total Sales: Actual vs. Fitted"
)
lines(train_agg_with_lags$time, sales_model$fitted.values, col = "red")
legend(
  "topleft", 
  c("Actual", "Fitted"), 
  col = c("green", "red"), 
  lwd = c(2, 2)
)

# Plot residuals over time
plot(
  train_agg_with_lags$time, 
  sales_model$residuals, 
  type = "l", 
  xlab = "Time (Months since Start)", 
  ylab = "Residuals", 
  main = "Residuals of Log Sales Regression"
)

# ACF plot of residuals
acf(
  sales_model$residuals, 
  main = "Autocorrelation of Residuals"
)


### K-Means Clustering for Store Location
# Prepare store data with unique locations
store_data <- train %>%
  distinct(store_nbr, city, state)

# Convert categorical variables to numeric (one-hot encoding)
store_matrix <- model.matrix(~ city + state - 1, data = store_data)

# Apply K-means clustering with optimal K
k_optimal <- 3
set.seed(123)
kmeans_result <- kmeans(store_matrix, centers = k_optimal, nstart = 20)

# Add cluster labels to store data
store_data$cluster <- as.factor(kmeans_result$cluster)

ggplot(store_data, aes(x = city, y = state, color = cluster)) +
  geom_jitter(size = 3, alpha = 0.7) +
  labs(title = "K-Means Clustering of Stores",
       x = "City", y = "State", color = "Cluster") +
  theme_minimal()



