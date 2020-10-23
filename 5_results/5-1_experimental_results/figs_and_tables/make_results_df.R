# creating tables and figures for results
library(tidyverse)

#dataset = "spooky"
dataset = "reuters"

 if (dataset == "reuters"){
   all_results_dirs <- list.dirs("latest_results/reuters/", full.names = T)[-1]
 }
 if (dataset == "spooky"){
   all_results_dirs <- list.dirs("latest_results/spooky/", full.names = T)[-1]
 }

all_results_dirs[1]

settings <- paste0(all_results_dirs[1], "//all_settings.csv")
results_df <- read_csv(settings, col_types = cols(
  .default = col_double(),
  stopwords = col_logical(),
  prune_vocab = col_double(),
  tf_idf = col_logical(),
  to_lower = col_logical(),
  remove_numbers = col_logical(),
  stemming = col_logical(),
  kernel_name = col_character(),
  kernel = col_character(),
  type = col_character(),
  dataset = col_character()
)) 

results_df$dataset

#remove first obs since it is used as template for results_df
all_results_dirs <- all_results_dirs[-1]
#grep("ngram__", all_results_dirs ) %>% length()

for (i in grep("ngram__", all_results_dirs )) {
  print(i)
  print(all_results_dirs[i])
  settings <- paste0(all_results_dirs[i], "//all_settings.csv")
  
  this_result <- read_csv(settings, col_types = cols(
    .default = col_double(),
    stopwords = col_logical(),
    prune_vocab = col_double(),
    tf_idf = col_logical(),
    to_lower = col_logical(),
    remove_numbers = col_logical(),
    stemming = col_logical(),
    kernel_name = col_character(),
    kernel = col_character(),
    type = col_character(),
    dataset = col_character()
  )) 
  results_df <- bind_rows(results_df, this_result)
  

}
#results_df_spooky <- results_df
#save(results_df_spooky, file = "results_df_spooky.RData")
#results_df_reuters <- results_df
#save(results_df_reuters, file = "results_df_reuters.RData")

