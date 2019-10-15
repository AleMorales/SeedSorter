# Add a classifier to mlr that will cluster the data into two groups (without peaking into Class)

#' @export
makeRLearner.classif.kmeans = function() {
  mlr::makeRLearnerClassif(
    cl = "classif.kmeans",
    package = c("stats", "clue"),
    par.set = makeParamSet(
      makeDiscreteLearnerParam(id = "algorithm", values = c("Hartigan-Wong", "Lloyd",
                               "Forgy", "MacQueen"), default = "Hartigan-Wong",
                               tunable = FALSE),
      makeLogicalLearnerParam(id = "trace", tunable = FALSE),
      makeLogicalLearnerParam(id = "seed_minority", default = TRUE, tunable = FALSE)
    ),
    par.vals = list(seed_minority = TRUE),
    properties = c("twoclass","numerics"),
    name = "k-means classifier",
    short.name = "kmeans",
    note = "Oversampling should not be applied.",
    callees = c("kmeans", "cl_predict")
  )
}

#' @export
trainLearner.classif.kmeans = function(.learner, .task, .subset, .weights = NULL, ...) {
  data = mlr::getTaskData(.task, .subset)
  stats::kmeans(dplyr::select(data, -Class), centers = 2, iter.max = 1000L)
}

#' @export
predictLearner.classif.kmeans = function(.learner, .model, .newdata, ...) {
    prediction = as.integer(clue::cl_predict(.model$learner.model, newdata = .newdata, type = "class_ids", ...))
    freqs = table(prediction)
    if(.learner$par.vals$seed_minority)
      prediction = factor(prediction, levels = c(which.min(freqs), which.max(freqs)), labels = c("S",'W'))
    else
      prediction = factor(prediction, levels = c(which.min(freqs), which.max(freqs)), labels = c("W",'S'))
}


