getwd()
# install.packages("reticulate")
library("reticulate")

# Set the path to the Python executable file
use_python("C:\\Users\\Emil\\anaconda3\\python.exe", required = T)
#use_python("C:\\ProgramData\\Anaconda3\\python.exe", required = T)

py_config()

dataset = "reuters"
#dataset = "spooky"
# settings_text_preprocess <- list(
#   to_lower = T,
#   remove_numbers = F,
#   stemming = F
# )
# # 0.0001 = 0.5e-04
# # prune_vocab = 0.01
# settings_preprocess <- list(
#   stopwords = TRUE,
#   ngram_range = c(1,1), # a range (a,b). if a=b, then its a n-gram where n=a. if a!=b, then its a combination of all n-grams in the range a to b.
#   prune_vocab = 0.001, # choose a doc_proportion_min range to prune vocabulary (reduce number of features), otherwise don't prune.
#   tf_idf = TRUE      # if use tf_idf weighting,
# )
kernel_settings = list(
  kernel_name = "linear"
)
settings_train = list(
  kernel = "precomputed",
  type = "C-svc"
)

bools = c(TRUE, FALSE)
ngrams = list(c(1,1), c(2,2), c(3,3))
prunes = c(NA, 
           1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03,
           5e-04)

160+160+(16*5) #400 configs


# ngrams[[1]]
# 
# length(ngrams)
# for(i in 1:length(ngrams)){
#   ngrams[[i]] %>% print()
# }
# 
# 
# bools[1]
# bools[2]
is.na(prunes) == FALSE 
(ngrams[[1]][1] == 1 && ngrams[[1]][2] == 1)

for (stops in bools){
  for (remnum in bools){
    for (stems in bools){
      for (tf_idf in bools){
        for (n in 1:length(ngrams)){
          for ( p in prunes){
            
            settings_text_preprocess <- list(
              to_lower = T,
              remove_numbers = remnum,
              stemming = stems
            )
            # 0.0001 = 0.5e-04
            # prune_vocab = 0.01
            settings_preprocess <- list(
              stopwords = stops,
              ngram_range = ngrams[[n]], # a range (a,b). if a=b, then its a n-gram where n=a. if a!=b, then its a combination of all n-grams in the range a to b.
              prune_vocab = p, # choose a doc_proportion_min range to prune vocabulary (reduce number of features), otherwise don't prune.
              tf_idf = tf_idf      # if use tf_idf weighting,
            )
            
            
            
            
            
            source("pre_train.R")
            
            id_name <- paste("ngram", settings_preprocess$ngram_range %>% paste0(collapse="_") ,
                             "k_name", kernel_settings$kernel_name,
                             "k", settings_train$kernel,
                             "typ", settings_train$type,
                             "dim", settings_preprocess$dimension_dtm_train[[2]],
                             "prune", settings_preprocess$prune_vocab,
                             "remNumb", settings_text_preprocess$remove_numbers,
                             "tolow", settings_text_preprocess$to_lower,
                             "stem", settings_text_preprocess$stemming,
                             "tfidf", settings_preprocess$tf_idf,
                             "stop", settings_preprocess$stopwords,
                             sep = "__")
            print("current:")
            print(id_name)
            
            print(settings_preprocess$dimension_dtm_train[2])
            if ( settings_preprocess$dimension_dtm_train[2] < 100 ){
              print("zero features or less than hundred, skipping")
              next
            }
             if (ngrams[[n]][1] == 1 && ngrams[[n]][2] == 1 && p %in% c(5e-04)){
               print(p)
               print("skipping pruning for unigrams")
               next
             }
            if (ngrams[[n]][1] == 2 && ngrams[[n]][2] == 2 && p %in% c(5e-04)){
              print(p)
              print("skipping pruning  for bigrams")
              next
            }
            if (ngrams[[n]][1] == 3 && ngrams[[n]][2] == 3 && p %in% c(4e-03, 5e-03, 6e-03, 7e-03, 8e-03)){
              print(p)
              print("skipping pruning for trigrams")
              next
            }
            
            
            system("python svm2.py")
            source("save_results.R")
            print("done")
          }
        }
      }
    }
  }
}




dataset = "reuters"
settings_text_preprocess <- list(
  to_lower = T,
  remove_numbers = F,
  stemming = F
)
# 0.0001 = 0.5e-04
# prune_vocab = 0.01
settings_preprocess <- list(
  stopwords = TRUE,
  ngram_range = c(1,1), # a range (a,b). if a=b, then its a n-gram where n=a. if a!=b, then its a combination of all n-grams in the range a to b.
  prune_vocab = NA, # choose a doc_proportion_min range to prune vocabulary (reduce number of features), otherwise don't prune.
  tf_idf = TRUE      # if use tf_idf weighting,
)



 source("pre_train.R")
 print(settings_preprocess$dimension_dtm_train[2])
 
 system("python svm2.py")
 py_run_file("svm2.py")
 
 source("save_results.R")
