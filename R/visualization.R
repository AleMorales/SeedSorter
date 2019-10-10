
# To make sure that the labels of the figures are humand readable
pretty_names = c(classif.xgboost = "Extreme Gradient Boosting",
                 classif.ksvm.preproc = "Support Vector Machine",
                 classif.randomForestSRC = "Random Forest",
                 classif.fnn.preproc = "K-Nearest Neighbours",
                 classif.qda.preproc  = "Quadratic Discriminant Analysis",
                 classif.naiveBayes = "Naive Bayes",
                 classif.lda = "Linear Discriminant Analysis",
                 classif.glmnet.preproc = "Logistic Regresion Elastic Net",
                 classif.extinction = "Extinction threshold",
                 classif.xgboost.tuned = "Extreme Gradient Boosting",
                 classif.ksvm.preproc.tuned = "Support Vector Machine",
                 classif.fnn.preproc.tuned = "K-Nearest Neighbours",
                 classif.glmnet.preproc.tuned = "Logistic Regresion Elastic Net")


#' Generate a boxplot with the performance score of the algorithms
#'
#' @param comparison The result of a comparison among algorithms as genearted by functions [compareAlgorithms()] or
#' [mergeComparisons()].
#'
#' @details Each datum corresponds to the prediction balanced error rate of a particular algorithm for a particular
#' prediction task (either from resampling within one task or for different tasks, see [compareAlgorithms()]).
#'
#' @return A `ggplot` object.
#' @export
plotComparison = function(comparison) {

  # Calculate performances
  perf = mlr::getBMRPerformances(comparison, as.df = TRUE)  %>%
            dplyr::mutate(learner.id = as.character(learner.id),
                          method = pretty_names[learner.id])

  # Generate the plot
  plot = ggplot2::ggplot(perf, ggplot2::aes(x = forcats::fct_reorder(method, ber), y = ber*100)) +
          ggplot2::geom_boxplot() +
          ggplot2::geom_jitter(width = 0.3, alpha = 0.5) +
          ggplot2::coord_flip() +
          ggplot2::xlab("") +
          ggplot2::ylab('Balanced error rate (%)') +
          ggplot2::theme_classic()

  return(plot)
}

