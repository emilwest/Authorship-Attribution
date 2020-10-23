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

# source reuters data set and features and y label
if (dataset == "reuters"){
  source("load_reuters.R") 
}
if (dataset == "spooky"){
  source("load_spooky.R")
}

source("preprocessing_best.R") 

# tutorials: 
# https://srdas.github.io/MLBook/Text2Vec.html 
# https://rstudio-pubs-static.s3.amazonaws.com/235421_0682227f43eb4f7294778dee0a7f3a10.html # preprocessing ideas
# http://text2vec.org/vectorization.html#basic_transformations
# https://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
# http://members.cbio.mines-paristech.fr/~jvert/svn/tutorials/practical/makekernel/makekernel_notes.pdf
# https://stackoverflow.com/questions/33813972/kernlab-kraziness-inconsistent-results-for-identical-problems
# https://radimrehurek.com/gensim/similarities/docsim.html#gensim.similarities.docsim.MatrixSimilarity


# articles
# https://www.aclweb.org/anthology/C18-1234.pdf # authorship reuters, de har accuracy på runt 65%
# https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf 69% acc on reuters with neural network
# https://scholarworks.iupui.edu/bitstream/handle/1805/15938/abdulmecits-purdue-thesis.pdf?sequence=1 #reuters


# kaggle 
# https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author/notebook

#sessionInfo() # for showing version numbers of all libraries



train_tokens <- text_preprocessing(train$text, settings_text_preprocess)
test_tokens <- text_preprocessing(test$text, settings_text_preprocess)
#all_tokens <- text_preprocessing(df$text, settings_text_preprocess)




preprocessed <- preprocess(train_tokens =  train_tokens, test_tokens  = test_tokens, settings_preprocess)
#preprocessed <- preprocess(train_tokens =  all_tokens, test_tokens  = test_tokens, settings_preprocess)

# holdout <- 2501:5000
# dtm <- preprocessed$dtm_train
# dtm_train <- dtm[-holdout,]
# dtm_test <- dtm[holdout,]
# tfidf = TfIdf$new()
# dtm_train = fit_transform(dtm_train, tfidf)
# dtm_test = transform(dtm_test, tfidf)
# sum(dtm_train[1,])


dtm_train <- preprocessed$dtm_train
dtm_test <- preprocessed$dtm_test

# normalize document term matrices with L1 norm ie rowsum = 1: sum(dtm_train[1,])==1
#dtm_train <- normalize(dtm_train, norm = "l1")
#dtm_test <- normalize(dtm_test, norm = "l1")

vocab <- preprocessed$vocabulary

# add dimension of dtms to settings list
settings_preprocess[["dimension_dtm_train"]] <- dim(dtm_train)
settings_preprocess[["dimension_dtm_test"]] <- dim(dtm_test)

settings_preprocess
settings_text_preprocess

####################################################
# top most frequent terms
# descriptive stats
top_terms <- vocab %>% select(term, term_count) %>% arrange(desc(term_count)) %>% top_n(term_count,n = 10)
top_terms

# plot of most frequent terms
top_terms %>% 
  mutate(term = fct_reorder(term, term_count)) %>%
  ggplot(aes(x=term, y=term_count)) +
  geom_bar(stat="identity") +
  labs(
    #title="Most frequent terms in document term matrix dictionary",
    y = "Term frequency",
    x = "Term") +
  coord_flip()

#####################################################

# https://stackoverflow.com/questions/26391367/how-to-use-string-kernels-in-scikit-learn
# string kernel
#?kernlab::ksvm(kernel = "stringdot", kpar = list(length= 3))


save( dtm_train  , file = "dtm_train.RData")
save( dtm_test  , file = "dtm_test.RData")




