# creating tables and figures for results
library(tidyverse)

# if (dataset == "reuters"){
#   all_results_dirs <- list.dirs("latest_results/reuters/", full.names = T)[-1]
# }
# if (dataset == "spooky"){
#   all_results_dirs <- list.dirs("latest_results/spooky/", full.names = T)[-1]
# }

all_results_dirs <- list.dirs("latest_results/reuters/", full.names = T)[-1]

dummy <- read_csv(paste0(all_results_dirs[1], "//all_settings.csv")) %>% t()
results_df <- matrix(NA, nrow=0, ncol=ncol(dummy)) %>% as.data.frame()
colnames(results_df) <- dummy[1,]
results_df

# store all results in single data fram by iterating through the folders in latest_results/
for (i in grep("ngram__", all_results_dirs ) ){
  #print(i)
  #print(all_results_dirs[i])
  settings <- paste0(all_results_dirs[i], "//all_settings.csv")
  dummy <- read_csv(settings) 
  #colnames(dummy) <- c("variable","value")
  #if (i %in% contains_stop_setting ==F){
  #  dummy <- dummy %>% add_row(variable = "stop", value = "TRUE", .before = 1)
  #}
  dummy <- t(dummy)
  results_df[i, ] <- dummy[2,]  # results_df[i, ] works only for data.frame and not tibbles
}

# now convert to tibble and parse columns with type_convert
results_df <- results_df %>% as_tibble() %>% type_convert()
results_df



#save(results_df, file = "results_df_reuterslatest.RData")
# Table 5
results_df %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name) %>%
  kable(format = "latex",
        caption = "Performance for different configurations for the Kaggle and Reuters data sets", 
        label = "tab:semlinear",
        col.names = c("remove stopwords" , "ngram ", "prune" ,
                      "tf-idf" , "dimensiondtm" , "tolower",
                       "remove numbers" , "stemming" , "kernel" ,
                      "accuracy" , "f1 macro" , "precision macro" , "recall macro" , "C"),
        booktabs=T) %>%
  kable_styling(latex_options = "striped") %>%
  pack_rows("Reuters", 1, 3, latex_gap_space = "0.5em") 
  #pack_rows("Kaggle", 4, 5, latex_gap_space = "0.5em") 

  

