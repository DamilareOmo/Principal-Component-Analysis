# unzip_project.R
# Milestone Assignment 2: Principal Component Analysis
# Anderson Cancer Center
#
# This R script unzips the submitted project folder
# (PCA_Milestone2.zip) so that the reviewer
# can access the Python script, README, and generated outputs.
#
# Usage:
#   1. Place this script in the same directory as the zip file.
#   2. Open R or RStudio and set your working directory to that folder,
#      e.g. setwd("path/to/folder")
#   3. Run: source("unzip_project.R")

# Name of the zip file to extract
zip_file_name <- "PCA_Milestone2.zip"

# Folder where contents will be extracted
output_directory <- "pca_project_extracted"

# Function to unzip the project folder, with error handling
unzip_project <- function(zip_path, destination) {
  if (!file.exists(zip_path)) {
    stop(paste("Error: Zip file not found ->", zip_path))
  }

  tryCatch({
    unzip(zipfile = zip_path, exdir = destination)
    message(paste("Successfully extracted '", zip_path,
                  "' to '", destination, "'", sep = ""))
  }, error = function(error_message) {
    message(paste("Failed to unzip file:", error_message))
  })
}

# Run the unzip function
unzip_project(zip_file_name, output_directory)

# List the extracted files so the user can confirm the contents
if (dir.exists(output_directory)) {
  extracted_files <- list.files(output_directory, recursive = TRUE)
  cat("\nExtracted files:\n")
  for (file in extracted_files) {
    cat(" -", file, "\n")
  }
}
