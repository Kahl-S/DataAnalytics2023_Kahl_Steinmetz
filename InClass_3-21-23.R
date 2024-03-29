# We can use the poly() function to estimate test error for the quadratic and cubic regression.
# Quadratic regression line
lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train) # Quadratic
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
# Cubic regression line
lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train) # Cubic
mean((mpg-predict(lm.fit3,Auto))[-train]^2)
# The error rates are: 19.82 for quadratics and 19.78 for cubic
# If we choose different training set instead, then we will obtain somewhat different errors,
# on the validation set.
set.seed(2)
train = sample(392,196)
lm.fit <- lm(mpg~horsepower, data = Auto, subset = train)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
# the error rate is 23.29
lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train) # Quadratic
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
# the error rate is 18.90
lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train) # Cubic
mean((mpg-predict(lm.fit3,Auto))[-train]^2)
# the error rate is 19.25
# Using this split of the observations into a training set and validation set,
# we find that the validation set error rates for the models with linear, quadratic,
# and cubic terms are 23.29, 18.90 and 19.25 respectively.
# The model that predict mpg using a quadratic function of horsepower performs better,
# than a models that only involves only a linear function of horsepower, and there is a,
# little evidence in favor of a model that uses a cubic function of horsepower.

# k-Fold Cross Validation
# The cv.glm() function can also be used to implement k-fold CV.
# We once again, set a random seed and initialize a vector in which,
# we will store the CV errors corresponding to the polynomial fits of orders one to #ten.
# here the K =10
# Read the cv.glm documentation
??cv.glm
set.seed(17)
help("rep") # read the documentation for the rep() function in R.
cv.error.10 = rep(0,10) # read documentation, help("rep")
for(i in 1:10){
  glm.fit = glm(mpg ~ poly(horsepower, i), data = Auto)
  cv.error.10[i] = cv.glm(Auto,glm.fit, K=10) $delta[1]
}
cv.error.10
# Notice the computation time is much shorter than LOOCV! :),
# This depends on your laptop performance :)
# We still see little evidence that using cubic or higher-order polynomials terms,
# leads to lower test error than simply using a quadratics fit.