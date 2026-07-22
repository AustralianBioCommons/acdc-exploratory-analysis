# Install and load the package
if (!require("remotes")) install.packages("remotes")
remotes::install_github("AustralianBioCommons/gen3-metadata", subdir = "gen3metadata-R")

library(gen3metadata)
library(ggplot2)
library(dplyr)

# Fetch metadata for the AusDiab project
result <- fetch_all_metadata("/home/rstudio/key.txt", "program1", "AusDiab")

# Extract the demographic list
demo_list <- result$demographic


# Flatten list of records into one data frame
demo_df <- bind_rows(demo_list)

# Age distribution
age_plot <- ggplot(demo_df, aes(x = baseline_age)) +
  geom_histogram(binwidth = 5)

# Education bar plot
education_plot <- ggplot(demo_df, aes(x = education)) +
  geom_bar()

age_plot
education_plot