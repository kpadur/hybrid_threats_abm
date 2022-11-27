
library(stats)
library(car)
library(olsrr)
library(lmtest)
library(dplyr)
data <- read.csv("...", header=TRUE)
head(data)
dat <- data
head(dat)

# Check correlation
cor(dat$cyber_beginning, dat$impact_use)
cor(dat$cyber_length, dat$impact_use)
cor(dat$misinfo_beginning, dat$impact_use)
cor(dat$misinfo_length, dat$impact_use)
cor(dat$combined_beginning, dat$impact_use)
cor(dat$combined_length, dat$impact_use)

cor(dat$cyber_beginning, dat$impact_pos_op)
cor(dat$cyber_length, dat$impact_pos_op)
cor(dat$misinfo_beginning, dat$impact_pos_op)
cor(dat$misinfo_length, dat$impact_pos_op)
cor(dat$combined_beginning, dat$impact_pos_op)
cor(dat$combined_length, dat$impact_pos_op)

cor(dat$cyber_beginning, dat$impact_neg_op)
cor(dat$cyber_length, dat$impact_neg_op)
cor(dat$misinfo_beginning, dat$impact_neg_op)
cor(dat$misinfo_length, dat$impact_neg_op)
cor(dat$combined_beginning, dat$impact_neg_op)
cor(dat$combined_length, dat$impact_neg_op)

# Cyber
model <- lm(impact_use~cyber_length+cyber_beginning, data = dat)
model <- lm(impact_pos_op~cyber_beginning, data = dat)
model <- lm(impact_neg_op~cyber_beginning, data = dat)
# Coordinated
model <- lm(impact_use~cyber_length + misinfo_length, data = dat)
model <- lm(impact_use~combined_length , data = dat)
model <- lm(impact_pos_op~cyber_length + misinfo_length, data = dat)
model <- lm(impact_pos_op~combined_length, data = dat)
model <- lm(impact_neg_op~cyber_length+misinfo_length, data = dat)
model <- lm(impact_neg_op~combined_length, data = dat)
# Model summary
summary(model)

# Residual Analysis
# Overview of the current model residuals
par(mfrow = c(2, 2))
plot(model)
# 1. The relationship between the predictor (x) and the outcome (y) is assumed to be linear.
# Residuals vs Fitted: is used to check the assumptions of linearity. 
plot(model, 1)
# Plot the observed versus predicted values
plot(model$fitted.values, model$model$BMI) 
# 2. The error term ε has zero mean. It should be equally probable that the errors fall above and below the regression line.
mean(resid(model))
# 3. The error term ε has constant variance σ² (homoscedasticity or homogeneity of variance)
# Scale-Location: is used to check the homoscedasticity of residuals (constant variance of residuals). 
plot(model, which = 3)
# Breusch-Pagan test
bptest(model)
# 4. The errors are uncorrelated. The computation of standard errors relies on statistical independence.
plot(model$residuals)
# 5. The errors are normally distributed.
# Normal Q-Q: is used to check the normality of residuals assumption. 
plot(model, which = 2) # magnitude of the residuals
# Histogram of residuals
hist(model$residuals)
# Shapiro-Wilk normality test
shapiro.test(model$residuals)
# 6. Test collinearity
model_corr <- cor(dat %>% dplyr::select(cyber_beginning, cyber_length),
                         use = "pairwise.complete.obs")
corrplot::corrplot(model_corr)
