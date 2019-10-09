
# Read a profile file
readprofile = function(filename) {
  # Figure out ncol
  row1 = scan(filename, nlines = 1)
  # Read profile as matrix
  prof = matrix(scan(filename), ncol = length(row1), byrow = TRUE)
  # We only use one channel (and up to 4 may be present)
  if(grepl("ch0", filename)) {
    #prof = plyr::alply(prof, 2, function(x) x)
    prof = purrr::map(1:ncol(prof), ~ prof[,.x])
  } else {
    prof = purrr::map(seq(1,length(row1), by = 4), ~ prof[,.x])
    #prof = plyr::alply(prof[,seq(1,length(row1), by = 4)], 2, function(x) x)
  }
  # Remove he entries with 0 that are actually outside of the profile
  #prof = plyr::llply(prof, function(x) {x = x[-(1:5)]; x = x[x > 0]})
  prof = purrr::map(prof, trim_profile)
}

# Trim a profile so that we only keep non-zero values and remove the initial bit
trim_profile = function(x) {
  x = x[-(1:5)]
  x = x[x > 0]
}

# Fit a 8th degree polynomial to the profile of each particle in the file
# Extract from the polynomial a series of features that are used for classification
fitpoly = function(y) {
  L = length(y)
  if(L < 5) return(c(P = max(y), Px = 0, C = 0))
  P = max(y) - min(y)
  x = seq(0.5, L - 0.5, by = 1)
  ry = y - min(y)
  model = lm(ry~x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) +
               I(x^8))
  EllipArea = pi*P*L/2/2
  coefs = coef(model)
  Area = sum(coefs[1] + coefs[2]*x + coefs[3]*x^2+ coefs[4]*x^3 + coefs[5]*x^4 +
               coefs[6]*x^5 + coefs[7]*x^6+ coefs[8]*x^7+ coefs[9]*x^8)
  out = list(P = P, Px = (which.max(y) + 0.5)/L, C = Area/EllipArea)
}

#' Process a file with particle profiles and save a new file with features for
#' classification
#'
#' @param filename Name of the file with raw data from the flow cytometer
#'
#' @return The function does not return anything but rather an fst file (from the \code{fst} package)
#' is saved with all features extracted from the profile of each particle.
#' @export
#'
#' @examples
#' processFile("testdata.text")
processFile = function(filename) {
  cat("Processing file ", filename, "\n")
  prof = readprofile(filename)
  coefs = furrr::future_map_dfr(prof, fitpoly)
  fst::write_fst(coefs, sub(".txt",".fst",filename), compress = 100)
}


#' Process files with particle profiles in a given folder
#'
#' @param dir Directory where the files with particles profiles is located
#'
#' @return The function does not return anything but rather an fst file (from the \code{fst} package)
#' is saved for every file with profile data, in the same directory as where the profile data is
#' located.
#'
#' @export
#'
#' @examples
#' processFiles("rawdata")
processFiles = function(dir = getwd()) {
  curdir = getwd()
  setwd(dir)
  on.exit(setwd(curdir))
  future::plan(future::multiprocess)
  # Figure out which are the files in the directory that end in "prf.txt"
  all_files = list.files()
  # Choose those that contain the pattern "_prf.txt"
  files = grep("_prf.txt", all_files, fixed = TRUE, value = TRUE)
  # For each file, read the profile, fit poly and save the matrix in a fst file
  purrr::map(files, processFile)
  setwd(curdir)
  return(invisible())
}


#' Read the data associated to a sorted sample
#'
#' @param mainfile The path to the file with time of fly and other summary data
#' @param profile The path to the fst files that contains the features extracted from the profiles
#'
#' @return Returns a data.frame with all the features calculated from the data. Each row corresponds to a particle
#'
#' @examples
#' readData("main_file.txt", "profile.fst")
readData = function(mainfile, profile) {
  profile_data = fst::read_fst(profile) %>%
                    tibble::as_tibble(.)
  main_data = suppressWarnings(readr::read_delim(mainfile, delim = "\t")) %>%
                    tibble::as_tibble(.) %>%
                    dplyr::filter(!is.na(Id), !is.na(Time)) %>%
                    dplyr::select(-X27)
  all_data = dplyr::bind_cols(main_data[1:nrow(profile_data),], profile_data)
  dplyr::select(all_data, TOF, Extinction, Green, Yellow, Red, P, Px, C)
}

#' Read and process seed sample (with optional complementary waste sample for training)
#'
#' @param datadir
#' @param main_file
#' @param profile_file
#' @param main_waste_file
#' @param profile_waste_file
#' @param waste
#' @param minsize
#' @param maxsize
#'
#' @return A tibble
#' @export
getData = function(main_file, profile_file, datadir = "",
                   main_waste_file = NULL, profile_waste_file = NULL,
                   calibration = c(38.325, 0.185)) {

  # Retrieve data from the main_file and profile_file
  data = readData(file.path(datadir, main_file),
                  file.path(datadir, profile_file)) %>%
          dplyr::mutate(Class = "S")

  # If a waste file is added, append it to the data
  if(!is.null(main_waste_file)) {
    waste = readData(file.path(datadir, main_waste_file),
                     file.path(datadir, profile_waste_file)) %>%
            dplyr::mutate(Class = "W")
    data = dplyr::bind_rows(data, waste)
  }

  # Perform transformations (feature engineering)
  data = dplyr::mutate(data,
                Size =  calibration[2]*TOF + calibration[1],
                FluorTotal = (Green + Red + Yellow)/3,
                rGreen = ifelse(FluorTotal > 0, Green/FluorTotal, 0),
                rRed = ifelse(FluorTotal > 0, Red/FluorTotal, 0),
                rYellow = ifelse(FluorTotal > 0, Yellow/FluorTotal, 0)) %>%
    # Select features to be used
    dplyr::select(Extinction, rGreen, rYellow, P, Px, C, Size, Class)
}


#' Use k-means clustering to separate waste from seeds using a size threshold as initial guess
#'
#' @param data
#' @param guess
#'
#' @return
#' @export
cleanSample = function(data, guess = 200) {
  seedData = dplyr::filter(data, Class == "S") %>%
              dplyr::select(-Class)
  wasteData = dplyr::filter(data, Class == "W")
  naive_dust = dplyr::filter(seedData, Size < guess)
  naive_seed = dplyr::filter(seedData, Size > guess)
  centers_dust = colMeans(naive_dust)
  centers_seed = colMeans(naive_seed)
  clusters = kmeans(seedData, centers = rbind(centers_dust, centers_seed), iter.max = 1e3)
  seedData = dplyr::mutate(seedData, Class = ifelse(clusters$cluster == 2, "S", "W"))
  dplyr::bind_rows(seedData, wasteData)
}



#'Create classification task and apply under/oversampling
#'
#' @param data
#' @param id
#'
#' @return
#' @export
createTask = function(data, id = "cleanseeds") {
  task = mlr::makeClassifTask(id = id, data = data, target = "Class", positive = "S")
  ratio = dplyr::group_by(data, Class) %>% summarise(n = n()) %>% (function(x) max(x$n)/min(x$n))
  task = mlr::smote(task, rate = ratio)
  return(task)
}
