

#######################################################
# Store results from python

source("evaluation.R")

report  <- read_csv("report.csv")
confmat <- read.csv("confmat.csv", header = F, col.names = levels(y.test), row.names = levels(y.test) )

# alternative to using the report:
results <- metric_on_confmat( as.matrix(confmat) )
results_list <- list(accuracy=NA,
                     f1_macro=NA,
                     precision_macro=NA,
                     recall_macro=NA)

results_list$accuracy <- results$accuracy
results_list$f1_macro <- results$f1_macro
results_list$precision_macro <- results$precision_macro
results_list$recall_macro <- results$recall_macro

bestC = read_csv("bestC.csv") %>% select(bestC)


all_settings <- c(settings_preprocess,
                  settings_text_preprocess,
                  kernel_settings,
                  settings_train,
                  results_list,
                  bestC = bestC[[1]],
                  dataset = dataset)
all_settings
all_settings_df <- all_settings %>% unlist() %>% t() %>% as.data.frame() %>% as_tibble()



# make sure columns are correct type:
all_settings_df <- all_settings_df %>%  
  mutate_at(vars(stopwords, tf_idf, to_lower, remove_numbers, stemming), as.logical) %>%
  mutate_at(vars(dataset,type,kernel,kernel_name), as.character) 


double_cols <- c("ngram_range1","ngram_range2", "dimension_dtm_test1", "dimension_dtm_test2", "prune_vocab",
                                  "dimension_dtm_test1", "dimension_dtm_test2",
                                  "accuracy", "f1_macro", "recall_macro", "bestC", "precision_macro")

# convert from factor to double by first converting to characters (very tedious)
all_settings_df[ ,double_cols] <- lapply(all_settings_df[ ,double_cols], as.character )
all_settings_df[ ,double_cols] <- lapply(all_settings_df[ ,double_cols], as.double )




ngram_range_text <- all_settings$ngram_range %>% paste0(collapse="_")
all_settings$kernel_name

id_name <- paste("ngram", ngram_range_text,
                 "k_name", all_settings$kernel_name,
                 "k", all_settings$kernel,
                 "typ", all_settings$type,
                 "dim", all_settings$dimension_dtm_train[[2]],
                 "prune", all_settings$prune_vocab,
                 "C", all_settings$bestC,
                 "remNumb", settings_text_preprocess$remove_numbers,
                 "tolow", settings_text_preprocess$to_lower,
                 "stem", settings_text_preprocess$stemming,
                 "tfidf", settings_preprocess$tf_idf,
                 "stop", settings_preprocess$stopwords,
                 sep = "__")



if (dataset == "reuters"){
  id_dirname  <- paste0("latest_results/reuters/", id_name)
}
if (dataset == "spooky"){
  id_dirname  <- paste0("latest_results/spooky/", id_name)
}


dir.create(path = id_dirname)


save_results <- function(){
  write.csv(all_settings_df, file=paste0(id_dirname, "/all_settings.csv") )
  save(results, file=paste0(id_dirname, "/results.RData") )
  save(confmat, file=paste0(id_dirname, "/confmat.RData")  )
}
save_results()

results_list

# tttt <- read_csv(paste0(id_dirname, "/all_settings.csv"), col_types = cols(
#   .default = col_double(),
#   stopwords = col_logical(),
#   prune_vocab = col_logical(),
#   tf_idf = col_logical(),
#   to_lower = col_logical(),
#   remove_numbers = col_logical(),
#   stemming = col_logical(),
#   kernel_name = col_character(),
#   kernel = col_character(),
#   type = col_character(),
#   dataset = col_character()
# ))
# tttt
