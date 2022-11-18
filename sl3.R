library(sl3)
library(origami)
pracid_folds_children <- make_folds(dat_obs, V=5)

task_q <- make_sl3_Task(
  data = dat_obs,
  outcome = "Y_abx",
  covariates = c("A_remote", 'age', 'male', 'asthma')
)

lrnr_glm <- make_learner(Lrnr_glm, folds = pracid_folds_children)
lrnr_xgboost <- make_learner(Lrnr_xgboost, folds = pracid_folds_children)

stack <- Stack$new(
  lrnr_glm,  lrnr_xgboost
)

sl <- Lrnr_sl$new(learners = stack, metalearner = Lrnr_nnls$new())

sl_fit <- sl$train(task = task_q)

sl_preds <- sl_fit$predict(task = task_q)

dat_obs_A_0 <- dat_obs %>% 
  mutate(A_remote=0)
  
task_q_0 <- make_sl3_Task(
  data = dat_obs,
  outcome = "Y_abx",
  covariates = c("A_remote", 'age', 'male', 'asthma')
)

sl_preds_A_0 <- sl_fit$predict(task = task_q_0)

dat_obs_A_1 <- dat_obs %>% 
  mutate(A_remote=1)

task_q_1 <- make_sl3_Task(
  data = dat_obs,
  outcome = "Y_abx",
  covariates = c("A_remote", 'age', 'male', 'asthma')
)

sl_preds_A_1 <- sl_fit$predict(task = task_q_1)


