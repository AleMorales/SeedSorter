# Retrieve a learning algorithm
getLearner = function(algorithm = "xgboost", osw.rate = 10, predict.type = "response") {

  # Depending on the type of algorithm, add different pre-processing steps
  if(algorithm == "xgboost") {
    learner = mlr::makeLearner(cl = "classif.xgboost", predict.type =  predict.type,
                         par.vals = list(objective = "binary:logistic", nthread = 2)) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate)

  } else if(algorithm == "knn") {
    learner = mlr::makeLearner(cl = "classif.fnn", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate) %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if(algorithm == "lda") {
    learner = mlr::makeLearner(cl = "classif.lda", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate)

  } else if(algorithm == "qda") {
    learner = mlr::makeLearner(cl = "classif.qda", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate) %>%
              mlr::makePreprocWrapperCaret(ppc.pca = TRUE)

  } else if(algorithm == "naiveBayes") {
    learner = mlr::makeLearner(cl = "classif.naiveBayes", predict.type =  predict.type) %>%
      mlr::makeOversampleWrapper(osw.rate = osw.rate)

  } else if(algorithm == "logistic") {
    learner = mlr::makeLearner(cl = "classif.glmnet", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate) %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if (algorithm == "svm") {
    learner = mlr::makeLearner(cl = "classif.ksvm", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate) %>%
              mlr::makePreprocWrapperCaret(method = c("center", "scale"))

  } else if (algorithm == "randomforest") {
    learner = mlr::makeLearner(cl = "classif.randomForestSRC", predict.type =  predict.type,
                               par.vals = list(ntree = 250))  %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate)

  } else if (algorithm == "extinction") {
    learner = mlr::makeLearner(cl = "classif.extinction", predict.type =  predict.type) %>%
              mlr::makeOversampleWrapper(osw.rate = osw.rate)

  } else if (algorithm == "kmeans") {
    learner = mlr::makeLearner(cl = "classif.kmeans", predict.type =  predict.type)

  } else {
    stop(paste0("Algorithm not supported: ", algorithm))
  }

  return(learner)
}

# BER metric for ensemble hill climbing algorithm
metricBER = function(pred, true) {
  pred = colnames(pred)[max.col(pred)] # Find prediction with maximum prob
  tb = table(true, pred) # Cross tabulation
  ber = 0.5*(tb[1,2]/(tb[1,1] + tb[1,2]) + tb[2,1]/(tb[2,1] + tb[2,2]))
  return(ber)
  #return(1 - sum(diag(tb)) / sum(tb))
}

# Retrieve a stacked ensemble
getEnsemble = function(algorithms = c("extinction", "naiveBayes", "lda", "qda"), osw.rate = 10) {
  # Retrieve the individual learners
  learners  = purrr::map(algorithms, getLearner, osw.rate = osw.rate, predict.type =  "prob")
  # Make stacked ensemble with hill.climb weights
  ensemble = mlr::makeStackedLearner(base.learners = learners, method = "hill.climb", predict.type = "prob",
                                parset = list(metric = metricBER))
}


# Retrieve a learning algorithm and put it into a wrapper for hyperparameter tuning
getTunedLearner = function(algorithm = "xgboost", osw.rate = 10, maxiter = 10L, lambda = 10L,
                           predict.type =  "response") {

  # Retrieve the basic learner
  learner = getLearner(algorithm, osw.rate, predict.type)

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
      ParamHelpers::makeIntegerParam("k", lower = 1, upper= 100))

  } else if(algorithm == "logistic") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("s", lower = 0, upper = 1),
      ParamHelpers::makeNumericParam("alpha", lower = 0, upper= 1))

  } else if(algorithm == "svm") {
    paramlist = ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("C", lower = -1, upper = 1, trafo = function(x) 10^x),
      ParamHelpers::makeNumericParam("sigma", lower = -0.5, upper = 1.5, trafo = function(x) 10^x))

  # Certain algorithms are simply not tuned, so we can skip the tuning wrapper...
  } else {
    return(learner)
  }

  # Resampling strategy for tuning is always 5-fold cv
  resampling = mlr::makeResampleDesc("CV",iters = 5, stratify = TRUE)

  # Measure to be minimized is balanced error rate
  measure = mlr::ber

  # By default the parameters are optimized with CMA-ES
  control = mlr::makeTuneControlCMAES(budget = maxiter*lambda, lambda = lambda)

  # Finally, wrap everything into an autotuning object
  tunedLearner = mlr::makeTuneWrapper(learner = learner, resampling = resampling,
                                 measures = measure, par.set = paramlist,
                                 control = control)
  return(tunedLearner)
}


# Retrieve a stacked ensemble with tuned learners
getTunedEnsemble = function(algorithms = c("extinction", "naiveBayes", "lda", "qda"),
                            osw.rate = 10, maxiter = 10L, lambda = 10L) {
  # Retrieve the individual learners
  learners  = purrr::map(algorithms, getTunedLearner, osw.rate = osw.rate, maxiter = maxiter,
                         lambda = lambda, predict.type = "prob")
  # Make stacked ensemble with hill.climb weights
  ensemble = mlr::makeStackedLearner(base.learners = learners, method = "hill.climb", predict.type = "prob",
                                     parset = list(metric = metricBER))
}

#' Train a learning algorithm and return the resulting model.
#'
#' @param algorithm Name of the algorithm to be used.
#' @param task A classification task as returned by [createTrainingTask()].
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
#' * `kmeans`: k-means clustering. This method will cluster the data into two clusters and assign
#' the minority cluster to seed or waste depending on the parameter `seed_minority`.
#'
#' @return A trained model that can be used to make predictions.
#' @export
trainAlgorithm = function(algorithm = "xgboost", task, osw.rate = 10) {

  learner = getLearner(algorithm, osw.rate)
  model = mlr::train(learner, task)
  return(model)
}

#' Train a stacked ensemble of learning algorithms and return the resulting model.
#'
#' @param algorithms Names of the algorithm to be used as a character vector.
#' @param task A classification task as returned by [createTrainingTask()].
#'
#' @details The list of algorithms that can be used are the same as for [trainAlgorithm()].
#' The stacked ensemble calculates a weight for each algorithm based on a hill climbing approach.
#'
#' @return A trained model that can be used to make predictions.
#' @export
trainEnsemble = function(algorithms = c("extinction", "naiveBayes", "lda", "qda"),
                         task, osw.rate = 10) {

  ensemble = getEnsemble(algorithms, osw.rate)
  model = mlr::train(ensemble, task)
  return(model)
}


#' Test a trained model with labelled data.
#'
#' @param model A trained model as returned by the function [trainAlgorithm()].
#' @param task A classification task as returned by [createTrainingTask()].
#'
#' @return A list with results of the prediction:
#'
#' * `prediction`: Results of the prediction of class [mlr::Prediction].
#'
#' * `error`: Values of the balanced error rate (`ber`) and the mean mis-classification error (`mmce`)
#' calculated from the prediction confusion matrix.
#'
#' @export
testModel = function(model, task) {

  data = mlr::getTaskData(task)
  prediction = predict(model, newdata = data)
  score = c(ber = mlr::performance(prediction, measures = mlr::ber),
            mmce = mlr::performance(prediction, measures = mlr::mmce))
  return(list(prediction = prediction, error = score))
}

#' Use a trained model to predict to classify a sample into seeds or non-seed particles.
#'
#' @param model A trained model as returned by the function [trainAlgorithm()].
#' @param data A dataset with all the features required by the algorithm.
#'
#' @return  Results of the prediction of class [mlr::Prediction].
#'
#' @export
classifySeeds = function(model, data) {
  prediction = predict(model, newdata = data)

}

#' Train an algorithm while tuning its hyperparameters and return the resulting model.
#'
#' @param algorithm Name of the algorithm to be used (same algorithms as in [trainAlgorithm()]).
#' @param task A classification task as returned by [createTrainingTask()].
#' @param maxiter Maximum number of iterations in the CMA-ES optimization of hyperparameters.
#' @param lambda Number of offspring in each iteration of the CMA-ES optimization of hyperparameters.
#' @param parallel Whether to use parallelization in the tuning of hyperparameters (default: `FALSE`).
#' @param nthreads Number of threads/workers to use for parallelization.
#'
#' @details The following algorithms can be tuned using CMA-ES optimization: `xgboost`, `logistic`,
#' `svm` and `knn`.
#'
#' @return A trained model that can be used to make predictions.
#' @export
tuneAlgorithm = function(algorithm = "xgboost", task, osw.rate = 10, maxiter = 10L, lambda = 10L,
                         parallel = FALSE, nthreads = parallel:::detectCores()) {

  learner = getTunedLearner(algorithm, osw.rate, maxiter, lambda)
  if(parallel) {
    parallelMap::parallelStart(mode = "socket", cpus = nthreads, level = "mlr.tuneParams")
    model = mlr::train(learner, task)
    parallelMap::parallelStop()
  } else {
    model = mlr::train(learner, task)
  }
  return(model)
}


#' Train a stacked ensemble of algorithms while tuning its hyperparameters and return the resulting model.
#'
#' @param algorithms Names of the algorithm to be used as a character vector
#' (same algorithms as in [trainAlgorithm()]).
#' @param task A classification task as returned by [createTrainingTask()].
#' @param maxiter Maximum number of iterations in the CMA-ES optimization of hyperparameters.
#' @param lambda Number of offspring in each iteration of the CMA-ES optimization of hyperparameters.
#' @param parallel Whether to use parallelization in the tuning of hyperparameters (default: `FALSE`).
#' @param nthreads Number of threads/workers to use for parallelization.
#'
#' @details The following algorithms can be tuned using CMA-ES optimization: `xgboost`, `logistic`,
#' `svm` and `knn`.
#'
#' @return A trained model that can be used to make predictions.
#' @export
tuneEnsemble = function(algorithms = c("extinction", "naiveBayes", "lda", "qda"),
                        task, osw.rate = 10, maxiter = 10L, lambda = 10L,
                        parallel = FALSE, nthreads = parallel:::detectCores()) {

  ensemble = getTunedEnsemble(algorithms, osw.rate, maxiter, lambda)
  if(parallel) {
    parallelMap::parallelStart(mode = "socket", cpus = nthreads, level = "mlr.tuneParams")
    model = mlr::train(ensemble, task)
    parallelMap::parallelStop()
  } else {
    model = mlr::train(ensemble, task)
  }
  return(model)
}


#' Compare performance of several algorithms on the same data, with or without hyperparameter tuning
#'
#' @param algorithms Vector with the names of algorithms to be compared (same algorithms as in [trainAlgorithm()]).
#' @param task Either one classification task for comparison using cross-validation or a list of tasks for
#'      comparisons across tasks (see Details).
#' @param tuning Whether to tune the learners or not (default: `FALSE`).
#' @param control Optional list of settings (see Details).
#'
#' @details The comparison of algorithms differs depending on where a single classification task or multiple
#' classification tasks are used. In the first approach, a repeated cross-validation scheme is used
#' to partition the task into subsets multiple times, resulting in a comparison for each combination of
#' subsets. If the algorithms are being tuned (which uses five-fold cross-validation), the resampling
#' using for this tuning is nested within the training folds of the outer cross-validation scheme.
#'
#' In the second approach, each learner is trained on each tasks (without resampling) and used to make
#' prediction on all other tasks. That is, if there are `n` tasks, this will result in `(n - 1)*n`
#' predictions, performed with `n` trained models.
#'
#' Parallelization is always applied over the outermost loop for a given learner. That is, when comparing
#' algorithms within one classification task, the parallization will be applied over the resampling
#' iterations of the outer cross-validation scheme. When comparing across tasks, the parallelization will
#' be applied over the tasks used for training the models.
#'
#' The following settings can be passed to the `control` argument:
#'
#' * `folds`: Number of cross-validation folds used in the outer resampling scheme when comparing algorithms
#' within one task It has no effect when comparing algorithms across multiple tasks. Default: 5.
#'
#' * `reps`: Number of repetitions of the cross-validation in the outer resampling scheme when comparing algorithms
#' within one task It has no effect when comparing algorithms across multiple tasks. Default: 3.
#'
#' * `parallel`: Whether to use parallelization or not. Default: `FALSE`.
#'
#' * `nthreads`: Number of threads/workers to be used for parallelization. Default is the number of cores as reported
#' by `parallel::detectCores()`.
#'
#' * `maxiter`: Maximum number of iterations in the CMA-ES optimization of hyperparameters. Default: 10.
#'
#' * `lambda`: Number of offspring in each iteration of the CMA-ES optimization of hyperparameters. Default: 10.
#'
#' * `seed`: Random seed used for resampling schemes.
#'
#' @return The result of the comparison, as an object of class [mlr::BenchmarkResult].
#' @export
compareAlgorithms = function(algorithms, task, tuning = FALSE, control = list()) {

  # Set control values
  defaultControl = list(folds = 5, reps = 1, parallel = FALSE,
                        nthreads = parallel::detectCores(),
                        maxiter = 10L, lambda = 10L,
                        seed = 2019, osw.rate = 10)
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
      learners[[i]] = getTunedLearner(algorithms[i], control$osw.rate, control$maxiter, control$lambda)
    else
      learners[[i]] = getLearner(algorithms[i], control$osw.rate)
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

#' Compare performance of several ensembles on the same data, with or without hyperparameter tuning
#'
#' @param algorithms Vector with lists of algorithms to be stacked (same algorithms as in [trainAlgorithm()]).
#' @param task Either one classification task for comparison using cross-validation or a list of tasks for
#'      comparisons across tasks (see Details).
#' @param tuning Whether to tune the learners or not (default: `FALSE`).
#' @param control Optional list of settings (see Details).
#'
#' @details See [compareAlgorithms()]. The argument `algorithms` should be assigned a list of vectors
#' (or list of lists). A stacked ensemble will be created for every element of the outer list.
#'
#' @return The result of the comparison, as an object of class [mlr::BenchmarkResult].
#' @export
compareEnsemble = function(algorithms, task, tuning = FALSE, control = list()) {

  # Set control values
  defaultControl = list(folds = 5, reps = 1, parallel = FALSE,
                        nthreads = parallel::detectCores(),
                        maxiter = 10L, lambda = 10L,
                        seed = 2019, osw.rate = 10)
  for(name in names(control)) {
    defaultControl[[name]] = control[[name]]
  }
  control = defaultControl

  # Setting the seed ensures the resampling of the data is the same
  set.seed(control$seed)

  # For each algorithm, retrieve the learner with pre-processing steps
  # In the case of tuning, retrieve the tuned wrapper
  ensembles = vector("list", length(algorithms))
  for(i in 1:length(algorithms)) {
    if(tuning) {
      ensembles[[i]] = getTunedEnsemble(algorithms[[i]], control$osw.rate, control$maxiter, control$lambda)
      ensembles[[i]]$id = paste0("Ensemble_",i)
    } else{
      ensembles[[i]] = getEnsemble(algorithms[[i]], control$osw.rate)
      ensembles[[i]]$id = paste0("Ensemble_",i)
    }
  }

  # If one task, then run canonical benchmark from mlr.
  # If multiple tasks, run special benchmark algorithms
  if(inherits(task, "Task")) {
    compareAlgorithmsInFile(ensembles, task, control)
  } else if (is.list(task)){
    compareAlgorithmsAcrossFiles(ensembles, task, control)
  } else {
    stop("Either provide a classification task or a list of classificationt tasks")
  }
}

# Run a benchmark from mlr
compareAlgorithmsInFile = function(learners, task, control) {
  # Resampling scheme is CV with user-defined folds
  if(control$reps >= 2)
    resampling = mlr::makeResampleDesc("RepCV", reps = control$reps,
                                       folds = control$folds, stratify = TRUE)
  else
    resampling = mlr::makeResampleDesc("CV", iters = control$folds, stratify = TRUE)

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
  results = purrr::map(learners, function(x) outermap(x, tasks))
  names(results) = purrr::map(learners, ~.x$id)
  result = structure(list(results = list(task = results), measures = list(mlr::ber),
                          learners = learners),
                     class = "BenchmarkResult")
}


#' Generate indices for combinations of tasks
#'
#' @param ntasks Number of tasks.
#'
#' @return A matrix with the indices that correspond to the tasks used for training and testing in a comparison
#' of algorithms across tasks. These indices can be used to figure out to which tasks a particular prediction
#' result belongs to.
#' @export
generateIndices = function(ntask) {
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
#' @param comparisons List of results obtained by calling \code{compareAlgorithms}.
#'
#' @details This will produce a new object that merges the results from the different comparisons. This is useful
#' when different settings are used to evaluate the performance of different algorithms.
#'
#' @return The result of merging the comparison objects, as an object of class [mlr::BenchmarkResult].
#' @export
mergeComparisons = function(comparisons) {
  mlr::mergeBenchmarkResults(as.list(comparisons))
}
