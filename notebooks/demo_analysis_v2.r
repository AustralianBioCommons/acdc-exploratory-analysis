if (!require("remotes")) install.packages("remotes")
remotes::install_github("AustralianBioCommons/gen3-metadata", subdir = "gen3metadata-R")

library(gen3metadata)
library(ggplot2)
library(dplyr)

# Fetch metadata for the AusDiab project
key <- '<insertapikey>'
result <- fetch_all_metadata(key, "program1", "synthetic_dataset_1")

# Extract the demographic list
demo_list <- result$agent_exposure


# Flatten list of records into one data frame
demo_df <- bind_rows(demo_list)

# bar plot
ggplot(demo_df, aes(x = agent_class)) +
  geom_bar()
