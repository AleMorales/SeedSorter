% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/processing.R
\name{readData}
\alias{readData}
\title{Read the data associated to a sorted sample}
\usage{
readData(mainfile, profile)
}
\arguments{
\item{mainfile}{The path to the file with time of fly and other summary data}

\item{profile}{The path to the fst files that contains the features extracted from the profiles}
}
\value{
Returns a data.frame with all the features calculated from the data. Each row corresponds to a particle
}
\description{
Read the data associated to a sorted sample
}
