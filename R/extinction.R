# Add a classifier to mlr that will find an optimum threshold for the Extinction feature to separate
# seeds from non-seeds

#' @export
makeRLearner.classif.extinction = function() {
  mlr::makeRLearnerClassif(
    cl = "classif.extinction",
    package = "mlr",
    par.set = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericLearnerParam(id = "threshold", lower = 0, upper = 1, tunable = FALSE,
                              when = "both", default = 0)
    ),
    properties = c("twoclass","numerics", "factors"),
    name = "Extinction threshold",
    short.name = "extinction",
    note = "Dataset must contain a column named Extinction"
  )
}

#' @export
trainLearner.classif.extinction = function(.learner, .task, .subset, .weights = NULL, ...) {
  data = mlr::getTaskData(.task, .subset)
  ext = median(data$Extinction)
  temp = optim(fn = extinction_ofun, par = ext, data = data, lower = 0, method = "Brent",
               upper = quantile(data$Extinction,prob = 0.9))
  .learner$par.vals$threshold = temp$par
  .learner
}

extinction_ofun = function(threshold, data) {
  predClass = as.factor(ifelse(data$Extinction > threshold, "S", "W"))
  ber = mlr::measureBER(truth = data$Class, response = predClass)
  ber
}


#' @export
predictLearner.classif.extinction = function(.learner, .model, .newdata, ...) {
  threshold = .model$learner.model$par.vals$threshold
  predictions = ifelse(.newdata$Extinction > threshold, "S", "W")
  if (.learner$predict.type == "response")
    return(as.factor(predictions))
  else {
    probs = matrix(NA, nrow = nrow(.newdata), ncol = 2)
    colnames(probs) = c("S", "W")
    probs[,1] = ifelse(predictions == "S", 1, 0)
    probs[,2] = ifelse(predictions == "W", 1, 0)
    return(probs)
  }
}


