% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/learning.R
\name{compareAlgorithms}
\alias{compareAlgorithms}
\title{Compare performance of several algorithms on the same data, with or without hyperparameter tuning}
\usage{
compareAlgorithms(algorithms, task, tuning = FALSE, control = list())
}
\arguments{
\item{algorithms}{Vector with the names of algorithms to be compared (same algorithms as in \code{\link[=trainAlgorithm]{trainAlgorithm()}}).}

\item{task}{Either one classification task for comparison using cross-validation or a list of tasks for
comparisons across tasks (see Details).}

\item{tuning}{Whether to tune the learners or not (default: \code{FALSE}).}

\item{control}{Optional list of settings (see Details).}
}
\value{
The result of the comparison, as an object of class \link[mlr:BenchmarkResult]{mlr::BenchmarkResult}.
}
\description{
Compare performance of several algorithms on the same data, with or without hyperparameter tuning
}
\details{
The comparison of algorithms differs depending on where a single classification task or multiple
classification tasks are used. In the first approach, a repeated cross-validation scheme is used
to partition the task into subsets multiple times, resulting in a comparison for each combination of
subsets. If the algorithms are being tuned (which uses five-fold cross-validation), the resampling
using for this tuning is nested within the training folds of the outer cross-validation scheme.

In the second approach, each learner is trained on each tasks (without resampling) and used to make
prediction on all other tasks. That is, if there are \code{n} tasks, this will result in \code{(n - 1)*n}
predictions, performed with \code{n} trained models.

Parallelization is always applied over the outermost loop for a given learner. That is, when comparing
algorithms within one classification task, the parallization will be applied over the resampling
iterations of the outer cross-validation scheme. When comparing across tasks, the parallelization will
be applied over the tasks used for training the models.

The following settings can be passed to the \code{control} argument:
\itemize{
\item \code{folds}: Number of cross-validation folds used in the outer resampling scheme when comparing algorithms
within one task It has no effect when comparing algorithms across multiple tasks. Default: 5.
\item \code{reps}: Number of repetitions of the cross-validation in the outer resampling scheme when comparing algorithms
within one task It has no effect when comparing algorithms across multiple tasks. Default: 3.
\item \code{parallel}: Whether to use parallelization or not. Default: \code{FALSE}.
\item \code{nthreads}: Number of threads/workers to be used for parallelization. Default is the number of cores as reported
by \code{parallel::detectCores()}.
\item \code{maxiter}: Maximum number of iterations in the CMA-ES optimization of hyperparameters. Default: 10.
\item \code{lambda}: Number of offspring in each iteration of the CMA-ES optimization of hyperparameters. Default: 10.
\item \code{seed}: Random seed used for resampling schemes.
}
}
