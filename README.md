# SeedSorter

[] `readData` should only return the columns that we actually need
[] Move `readData` to the other file
[] Do not save the intermediate fst file -> save instead the result of readData if an option to save copy is enable. Implement a cache based system to check if the source files are older than the corresponding fst file. If the source files are older, just load the saved copy, no need to process the data again!
[] Add a function to calculate `size` from TOF using a calibration matrix (internally it fits an rlm model to the calibration matrix)
[] Sieve the data given a maximum and minimum `size` (by default 0 and Inf, respectively)
[] Choose an algorithm and train it on a labelled dataset (internally uses caret)
[] Apply an algorithm to classify unlabelled data, given probability for each classification
[] Calculate empirical distributions of summary statistics and their standard errors using Monte Carlo on the probabilistic classification  -> Quantify the effect of uncertainty on the classification.
