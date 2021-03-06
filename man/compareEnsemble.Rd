% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/learning.R
\name{compareEnsemble}
\alias{compareEnsemble}
\title{Compare performance of several ensembles on the same data, with or without hyperparameter tuning}
\usage{
compareEnsemble(algorithms, task, tuning = FALSE, control = list())
}
\arguments{
\item{algorithms}{Vector with lists of algorithms to be stacked (same algorithms as in \code{\link[=trainAlgorithm]{trainAlgorithm()}}).}

\item{task}{Either one classification task for comparison using cross-validation or a list of tasks for
comparisons across tasks (see Details).}

\item{tuning}{Whether to tune the learners or not (default: \code{FALSE}).}

\item{control}{Optional list of settings (see Details).}
}
\value{
The result of the comparison, as an object of class \link[mlr:BenchmarkResult]{mlr::BenchmarkResult}.
}
\description{
Compare performance of several ensembles on the same data, with or without hyperparameter tuning
}
\details{
See \code{\link[=compareAlgorithms]{compareAlgorithms()}}. The argument \code{algorithms} should be assigned a list of vectors
(or list of lists). A stacked ensemble will be created for every element of the outer list.
}
