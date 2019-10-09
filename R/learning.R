# Retrieve a learning algorithm
getLearner = function(algorithm = "xgboost") {

  # Depending on the type of algorithm, add different pre-processing steps
  if(algorithm == "xgboost") {
    learner = mlr::makeLearner(cl = "classif.xgboost",
                         par.vals = list(objective = "binary:logistic", nthread = 2))

  } else if(algorithm == "knn") {
    learner = mlr::makeLearner(cl = "classif.fnn") %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if(algorithm == "lda") {
    learner = mlr::makeLearner(cl = "classif.lda")

  } else if(algorithm == "qda") {
    learner = mlr::makeLearner(cl = "classif.qda") %>%
              mlr::makePreprocWrapperCaret(ppc.pca = TRUE)

  } else if(algorithm == "naiveBayes") {
    learner = mlr::makeLearner(cl = "classif.naiveBayes")

  } else if(algorithm == "logistic") {
    learner = mlr::makeLearner(cl = "classif.glmnet", predict.type = "prob") %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if (algorithm == "svm") {
    learner = mlr::makeLearner(cl = "classif.ksvm") %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if (algorithm == "randomforest") {
    learner = mlr::makeLearner(cl = "classif.randomForestSRC", par.vals = list(ntree = 250))

  } else if (algorithm == "extinction") {
    learner = mlr::makeLearner(cl = "classif.extinction")

  } else {
    stop(paste0("Algorithm not supported: ", algorithm))
  }

  return(learner)
}

# Retrieve a learning algorithm and put it into a wrapper for hyperparameter
# tuning
getTunedLearner = function(algorithm = "xgboost", maxiter = 10L, lambda = 10L) {

  # Retrieve the basic learner
  learner = getLearner(algorithm)

  # Define the list of hyperparameters to be tuned depending on the algorithm
  if(algorithm == "xgboost") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("subsample", lower = 0, upper = 1),
      ParamHelpers::makeNumericParam("nrounds", lower = 0, upper = 2, trafo = function(x) trunc(10^x)),
      ParamHelpers::makeNumericParam("max_depth", lower = 0, upper = 2, trafo = function(x) trunc(10^x)),
      ParamHelpers::makeNumericParam("eta", lower = -2, upper = 0, trafo = function(x) 10^x),
      ParamHelpers::makeNumericParam("lambda", lower = -2, upper = 0, trafo = function(x) 10^x),
      ParamHelpers::makeNumericParam("alpha", lower = -2, upper = 0, trafo = function(x) 10^x))

  } else if (algorithm == "knn") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeIntegerParam("k", lower = 1, upper= 25))

  } else if(algorithm == "logistic") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("s", lower = 0, upper = 1),
      ParamHelpers::makeNumericParam("alpha", lower = 0, upper= 1))

  } else if(algorithm == "svm") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("C", lower = -1, upper = 1, trafo = function(x) 10^x),
      ParamHelpers::makeNumericParam("sigma", lower = -1, upper = 1, trafo = function(x) 10^x))

  # Certain algorithms are simply not tuned, so we can skip the tuning wrapper...
  } else {
    return(learner)
  }

  # Resampling strategy for tuning is always 5-fold cv
  resampling = mlr::makeResampleDesc("CV",iters = 5, stratify = TRUE)

  # Measure to be minimized is mean misclassification error
  measure = mlr::ber

  # By default the parameters are optimized with CMA-ES, except for
  # K-Nearest Neighbours where a grid system (with the same budget) is used
  if(algorithm == "knn") {
    control = mlr::makeTuneControlGrid(resolution = 10)
  } else {
    control = mlr::makeTuneControlCMAES(budget = maxiter*lambda, lambda = lambda)
  }

  # Finally, wrap everything into an autotuning object
  tunedLearner = mlr::makeTuneWrapper(learner = learner, resampling = resampling,
                                 measures = measure, par.set = paramlist,
                                 control = control)
  return(tunedLearner)
}

#' Train a learning algorithm and return the resulting model.
#'
#' @param algorithm Name of the algorithm to be used.
#' @param task A classification task as returned by [createTask()].
#'
#' @details The list of algorithms that can be used are:
#'
#' * `xgboost`: The XGBoost algorithm.
#'
#' * `knn`: The fast k-nearest neighbour algorithm fromt the `FNN` package.
#'
#' * `lda`: Linear discriminant analysis.
#'
#' * `qda`: Quadratic discriminant analysis applied to principal components.
#'
#' * `naiveBayes`: Naive Bayes classifier.
#'
#' * `logistic`: Logistic regression regularized with elasticnet grid penalty.
#'
#' * `svm`: Support vector machine with radial basis kernel (aka Gaussian kernel).
#'
#' * `randomForest`: Multithreaded random forest as implemented in the package `randomForestSRC`.
#'
#' @return A trained model that can be used to make predictions
#' @export
trainAlgorithm = function(algorithm = "xgboost", task) {

  learner = getLearner(algorithm)
  model = mlr::train(learner, task)
  return(model)
}


#' Use a trained model to predict to classify a sample into seeds or non-seed particles.
#'
#' @param model A trained model as returned by the function [trainAlgorithm()].
#' @param task A classification task as returned by [createTask()].
#'
#' @return A list with results of the prediction:
#'
#' * `prediction`: Results of the prediction of class [mlr::Prediction].
#'
#' * `error`: Values of the balanced error rate (`ber`) and the mean mis-classification error (`mmce`)
#' calculated from the prediction confusion matrix.
#'
#' @export
classifySeeds = function(model, task) {

  data = mlr::getTaskData(task)
  prediction = predict(model, newdata = data)
  score = c(ber = mlr::performance(prediction, measures = mlr::ber),
            mmce = mlr::performance(prediction, measures = mlr::mmce))
  return(list(prediction = prediction, error = score))
}


#' Train a learning algorithm while tuning its hyperparameters and return the resulting model.
#'
#' @param algorithm Name of the algorithm to be used (same as for [trainAlgorithm()]).
#' @param task A classification task as returned by [createTask()].
#' @param maxiter
#' @param parallel
#' @param nthreads
#'
#' @return
#' @export
#'
#' @examples
tuneAlgorithm = function(algorithm = "xgboost", task, maxiter = 10L, lambda = 10L,
                     parallel = FALSE, nthreads = parallel:::detectCores()) {

  learner = getTunedLearner(algorithm, maxiter, lambda)
  if(parallel) {
    parallelMap::parallelStart(mode = "socket", cpus = nthreads, level = "mlr.tuneParams")
    model = mlr::train(learner, task)
    parallelMap::parallelStop()
  } else {
    model = mlr::train(learner, task)
  }
  return(model)
}

#' Compare performance of several algorithms on the same data, with or without hyperparameter tuning
#'
#' @param algorithms
#' @param task
#' @param tuning
#' @param control
#'
#' @return
#' @export
compareAlgorithms = function(algorithms, task, tuning = FALSE, control = list()) {

  # Set control values
  defaultControl = list(folds = 5, reps = 1, parallel = FALSE,
                        nthreads = parallel:::detectCores(),
                        maxiter = 10L, lambda = 10L,
                        seed = 2019)
  for(name in names(control)) {
    defaultControl[[name]] = control[[name]]
  }
  control = defaultControl

  # Setting the seed ensures the resampling of the data is the same
  set.seed(control$seed)

  # For each algorithm, retrieve the learner with pre-processing steps
  # In the case of tuning, retrieve the tuned wrapper
  learners = vector("list", length(algorithms))
  for(i in 1:length(algorithms)) {
    if(tuning)
      learners[[i]] = getTunedLearner(algorithms[i], control$maxiter, control$lambda)
    else
      learners[[i]] = getLearner(algorithms[i])
  }

  # If one task, then run canonical benchmark from mlr.
  # If multiple tasks, run special benchmark algorithms
  if(inherits(task, "Task")) {
    compareAlgorithmsInFile(learners, task, control)
  } else if (is.list(task)){
    compareAlgorithmsAcrossFiles(learners, task, control)
  } else {
    stop("Either provide a classification task or a list of classificationt tasks")
  }
}

# Run a benchmark from mlr
compareAlgorithmsInFile = function(learners, task, control) {
  # Resampling scheme is CV with user-defined folds
  resampling = mlr::makeResampleDesc("RepCV", reps = control$reps,
                                     folds = control$folds, stratify = TRUE)

  # Run benchmark, optionally running resampling folds or tuning in parallel
  if(control$parallel) {
    parallelMap::parallelStart(mode = "socket", cpus = control$nthreads, level = "mlr.resample")
  }

  bmr = mlr::benchmark(learners = learners, tasks = task, resamplings = resampling,
                       measures = list(mlr::mmce, mlr::ber))

  if(control$parallel) parallelMap::parallelStop()

  return(bmr)
}

# Ad-hoc resampling where the algorithms are trained in one file and predict another
# It returns an incomplete BenchmarkResult object (just with the measures)
compareAlgorithmsAcrossFiles = function(learners, tasks, control) {
  # Generate all possible combinations of tasks
  #indices = generateIndices(tasks)
  # Switch between parallel and sequential plans
  if(control$parallel) {
    future::plan(future::multiprocess)
  } else {
    future::plan(future::sequential)
  }
  # Create the BenchmarkResult object
  results = furrr::future_map(learners, function(x) outermap(x, tasks))
  names(results) = purrr::map(learners, ~.x$id)
  result = structure(list(results = list(task = results), measures = list(mlr::ber),
                          learners = learners),
                     class = "BenchmarkResult")
}


#' Generate indices for combinations of tasks
#'
#' @param tasks
#'
#' @return
#' @export
generateIndices = function(tasks) {
  ntask = length(tasks)
  indices = cbind(rep(1:ntask, each = ntask), rep(1:ntask, times = ntask))
  indices = indices[which(indices[,1] != indices[,2]),]
  colnames(indices) = c("train", "test")
  indices
}

# Produce a ResampleResult for a given learner
outermap = function(learner, tasks) {
  # Outer loop over training task performed in parallel and the inner loop unnested
  results = furrr::future_map(1:length(tasks), ~innermap(learner, tasks, .x)) %>%
    unlist(recursive = FALSE)

  # Create the ResampleResult object
  task.id = NULL
  learner.id = learner$id
  task.desc = mlr::getTaskDesc(tasks[[1]])
  measures.train = data.frame(iter = 1:length(results),
                              ber = NA, mmce = NA)
  measures.test = data.frame(iter = 1:length(results),
                             ber = purrr::map_dbl(results, ~.x[[2]][1]),
                             mmce = purrr::map_dbl(results, ~.x[[2]][2]))
  aggr = c(ber.test.mean = mean(measures.test$ber),
           mmce.test.mean = mean(measures.test$mmce))
  models = NULL
  err.msgs = purrr::map(1:length(results), ~list())
  err.dumps = purrr::map(1:length(results), ~list())
  extract = purrr::map(1:length(results), ~NULL)
  runtime = NA
  learner = learner
  pred = purrr::map(results, ~.x[[1]])
  structure(list(learner.id = learner.id,
                 task.id = task.id,
                 task.desc = task.desc,
                 measures.train = measures.train,
                 measures.test = measures.test,
                 aggr = aggr,
                 pred = pred,
                 models = models,
                 err.msgs = err.msgs,
                 err.dumps = err.dumps,
                 extract = extract,
                 runtime = runtime,
                 learner = learner), class = "ResampleResult")
}

# Train the algorithm on a given task and use it to make predictions in the rest of the tasks
# This is the innermost loop in the comparison of algorithms across tasks
innermap = function(learner, tasks, outer) {
  # Train model on task determined by index outer
  model = mlr::train(learner, tasks[[outer]])
  # Loop over all tasks (except the one determined by outer) and make prediction
  results = purrr::map(c(1:length(tasks))[-outer], ~classifySeeds(model, tasks[[.x]]))
}


#' Merge results of multiple calls to \code{compareAlgorithms}
#'
#' @param comparisons
#'
#' @return
#' @export
#'
#' @examples
mergeComparisons = function(comparisons) {
  mlr::mergeBenchmarkResults(as.list(comparisons))
}
