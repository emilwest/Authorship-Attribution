# Main file
library(kernlab) # for svm 
library(doParallel) # for parallel computing
library(parallel) # for automatic detection of cores etc
library(caret) # svm alternative
library(SnowballC) # for stemming
#install.packages("text2vec")
library(text2vec) # for BoW model
library(tidyverse) 
#install.packages("tm")
library(tm) # for text preprocessing

dataset = "spooky"
source("load_spooky.R") 
source("best_kaggle_config.R")
train_tokens <- text_preprocessing(train$text, settings_text_preprocess)
test_tokens <- text_preprocessing(test$text, settings_text_preprocess)
preprocessed <- preprocess(train_tokens =  train_tokens, test_tokens  = test_tokens, settings_preprocess)
dtm_train <- preprocessed$dtm_train
dtm_test <- preprocessed$dtm_test
vocab <- preprocessed$vocabulary
allterms = vocab$term
save(allterms, file="vocab.RData") # for constructing semantic matrix in python
settings_preprocess[["dimension_dtm_train"]] <- dim(dtm_train)
settings_preprocess[["dimension_dtm_test"]] <- dim(dtm_test)
settings_preprocess$dimension_dtm_train
idf <- vocab %>% mutate(idf = log(1 + nrow(dtm_train)/doc_count )) 
R <- diag(idf$idf) %>% as("CsparseMatrix")


save( dtm_train  , file = "dtm_train.RData")
save( dtm_test  , file = "dtm_test.RData")
save(R, file="Rmatrix.RData")

# 1+2 gram  macro avg       0.69      0.66      0.67      2500
# 1+2 gram 7e-03, term_count_min = 10, dim=  macro avg       0.70      0.67      0.68










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
                  bestC = bestC[[1]] )
all_settings
all_settings_df <- all_settings %>% unlist() %>% as.data.frame()

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

id_name


# if (dataset == "reuters"){
#   id_dirname  <- paste0("latest_results/reuters/", id_name)
# }
# if (dataset == "spooky"){
#   id_dirname  <- paste0("latest_results/spooky/", id_name)
# }


if (dataset == "reuters"){
  id_dirname  <- paste0("latest_results/new/reuters/", id_name)
}
if (dataset == "spooky"){
  id_dirname  <- paste0("latest_results/new/spooky/", id_name)
}


getwd()
dir.create(path = id_dirname)


save_results <- function(){
  write.csv(all_settings_df, file=paste0(id_dirname, "/all_settings.csv") )
  save(results, file=paste0(id_dirname, "/results.RData") )
  save(confmat, file=paste0(id_dirname, "/confmat.RData")  )
}
save_results()

results_list










# # ROC CURVES
# preds_prob <- read.table("preds_prob.txt")
# preds_prob[1:10,1:10]
# #install.packages("multiROC")
# library(multiROC)
# ?multi_roc
# # A data frame is required for this function as input. 
# # This data frame should contains true label (0 - Negative, 1 - Positive) 
# # columns named as XX_true (e.g. S1_true, S2_true and S3_true) 
# # and predictive scores (continuous) columns named as XX_pred_YY (e.g. S1_pred_SVM, S2_pred_RF),
# # thus this function allows calcluating ROC on mulitiple classifiers.
# 
# rocdata <- as.data.frame(preds_prob)
# rocdata
# y.test
# model.ma
# # convert true labels to binary matrix and change colnames to format of multi_roc
# y.testdf <- as_tibble(y.test, rownames="id") %>% mutate(id = as.factor(id))
# yy <- sapply(levels(y.testdf$value), function(x) as.integer(x == y.testdf$value))
# colnames(yy) <- paste(colnames(yy), "true",  sep = "_")
# 
# 
# colnames(preds_prob) <- levels(y.test)
# colnames(preds_prob) <- paste(colnames(preds_prob),  "pred", "SVM", sep = "_")
# 
# 
# datroc <- cbind(yy, preds_prob)




