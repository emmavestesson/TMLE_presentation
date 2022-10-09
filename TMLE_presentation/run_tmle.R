library(tidyverse) # for data manipulation
library(SuperLearner) # for ensemble learning

set.seed(7) # for reproducible results

sl_libs <- c('SL.glmnet', 'SL.ranger') # a library of machine learning algorithms (penalized regression, random forests, and multivariate adaptive regression splines)

generate_data <- function(n){ 
  male <- rbinom(n, size=1, prob=0.4) # binary confounder
  asthma <- rbinom(n, size=1, prob=0.5) # binary confounder
  age <- round(runif(n, min=0, max=16)) # continuous confounder
  A_remote  <- rbinom(n, size=1, prob= plogis(-1 + 0.3*asthma + 0.01*age^2 +  0.2*male*asthma)) # binary treatment depends on confounders
  Y_abx <- rbinom(n, size=1, prob= plogis(-1 + A_remote - 0.1*age + 0.3*asthma + 0.3*male + sin(0.1*asthma*male))) # binary outcome depends on confounders
  return(tibble(Y_abx, male, age, asthma, A_remote))
}

n <- 1000
dat_obs <- generate_data(n) # generate a data set with n observations
dat_obs %>% gtsummary::tbl_summary(by = A_remote)

Y <- dat_obs$Y_abx
W_A <- dat_obs %>% select(-Y_abx) # remove the outcome to make a matrix of predictors (A, age, asthma, male, W4) for SuperLearner

### Step 1: Estimate Q
Q <- SuperLearner(Y = Y, # Y is the outcome vector
                  X = W_A, # W_A is the matrix of age, asthma, male,  and A
                  family=binomial(), # specify we have a binary outcome
                  SL.library = sl_libs) # specify our superlearner library of LASSO, RF, and MARS


Q_A <- as.vector(predict(Q)$pred) # obtain predictions for everyone using the treatment they actually received
W_A1 <- W_A %>% mutate(A_remote = 1)  # data set where everyone received treatment
Q_1 <- as.vector(predict(Q, newdata = W_A1)$pred) # predict on that everyone-exposed data set
W_A0 <- W_A %>% mutate(A_remote = 0) # data set where no one received treatment
Q_0 <- as.vector(predict(Q, newdata = W_A0)$pred)

dat_tmle <- tibble(Y_abx = dat_obs$Y_abx, A_remote = dat_obs$A_remote, Q_A, Q_0, Q_1)

dat_tmle <- dat_obs %>% 
  select(Y_abx, A_remote) %>% 
  bind_cols(Q_A=Q_A, Q_0 = Q_0, Q_1 = Q_1)
### Step 2: Estimate g and compute H(A,W)
A <- dat_obs$A_remote
W <- dat_obs %>% select(-Y_abx, -A_remote) # matrix of predictors that only contains the confounders age, asthma, male, and W4
g <- SuperLearner(Y = A, # outcome is the A (treatment) vector
                  X = W, # W is a matrix of predictors
                  family=binomial(), # treatment is a binomial outcome
                  SL.library=sl_libs) # using same candidate learners; could use different learners

g_w <- as.vector(predict(g)$pred) # Pr(A=1|W)
H_1 <- 1/g_w
H_0 <- -1/(1-g_w) # Pr(A=0|W) is 1-Pr(A=1|W)
dat_tmle <- # add clever covariate data to dat_tmle
  dat_tmle %>%
  bind_cols(
    H_1 = H_1,
    H_0 = H_0) %>%
  mutate(H_A = case_when(A_remote == 1 ~ H_1, # if A is 1 (treated), assign H_1
                         A_remote == 0 ~ H_0))  # if A is 0 (not treated), assign H_0

### Step 3: Estimate fluctuation parameter
glm_fit <- glm(Y ~ -1 + offset(qlogis(Q_A)) + H_A, data=dat_tmle, family=binomial) # fixed intercept logistic regression
eps <- coef(glm_fit) # save the only coefficient, called epsilon in TMLE lit

### Step 4: Update Q's
H_A <- dat_tmle$H_A # for cleaner code in Q_A_update
Q_A_update <- plogis(qlogis(Q_A) + eps*H_A) # updated expected outcome given treatment actually received
Q_1_update <- plogis(qlogis(Q_1) + eps*H_1) # updated expected outcome for everyone receiving treatment
Q_0_update <- plogis(qlogis(Q_0) + eps*H_0) # updated expected outcome for everyone not receiving treatment

### Step 5: Compute ATE
tmle_ate <- mean(Q_1_update - Q_0_update) # mean diff in updated expected outcome estimates

### Step 6: compute standard error, CIs and pvals
infl_fn <- (Y - Q_A_update) * H_A + Q_1_update - Q_0_update - tmle_ate # influence function
tmle_se <- sqrt(var(infl_fn)/nrow(dat_obs)) # standard error
conf_low <- tmle_ate - 1.96*tmle_se # 95% CI
conf_high <- tmle_ate + 1.96*tmle_se
pval <- 2 * (1 - pnorm(abs(tmle_ate / tmle_se))) # p-value at alpha .05

tmle_ate
conf_low
conf_high


saveRDS(dat_tmle, here::here('data', 'dat_tmle.RDS'))
saveRDS(Q, here::here('data', 'Q.RDS'))
saveRDS(g, here::here('data', 'g.RDS'))
saveRDS(W_A, here::here('data', 'W_A.RDS'))
saveRDS(dat_obs, here::here('data', 'dat_obs.RDS'))

