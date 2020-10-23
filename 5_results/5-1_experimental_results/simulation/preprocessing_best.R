# create a function that stores parameter settings etc


## PREPROCESSING SETTINGS ##

# prep_fun = tolower
# tok_fun = word_tokenizer
# stopwords <- tm::stopwords()
# ngram_range = c(1,1)
# prune_vocab = NA
# tf_idf = TRUE
#?word_tokenizer

# settings_preprocess <- list(
#   prep_fun = tolower,
#   tok_fun = word_tokenizer,
#   stopwords = tm::stopwords(),
#   ngram_range = c(1,1), # a range (a,b). if a=b, then its a n-gram where n=a. if a!=b, then its a combination of all n-grams in the range a to b.
#   prune_vocab = NA, # choose a doc_proportion_min range to prune vocabulary (reduce number of features), otherwise don't prune.
#   tf_idf = TRUE # if use tf_idf weighting 
# )

stem_tokenizer <-  function(x) {
  tokens = word_tokenizer(x)
  lapply(tokens, SnowballC::wordStem, language="en")
}
#?create_vocabulary

text_preprocessing <- function(text, ...){
  
  args <- list(...)[[1]]
  to_lower <- args$to_lower
  remove_numbers <- args$remove_numbers
  stemming <- args$stemming
  
  
  tok_fun <- text2vec::word_tokenizer
  
  if (to_lower == TRUE){
    text <- tolower(text)
  }
  if (remove_numbers == TRUE){
    text <- removeNumbers(text)
  }
  if (stemming == TRUE){
    tok_fun = stem_tokenizer
  }
  
  preprocessed_text <- tok_fun(text)
  
  return(preprocessed_text)
}


preprocess <- function(
  train_tokens = train_tokens, # train tokens
  test_tokens = test_tokens, # test tokens
  ...
){
  
  args <- list(...)[[1]]
  
  stopwords <- args$stopwords
  ngram_range = args$ngram_range
  prune_vocab = args$prune_vocab
  tf_idf = args$tf_idf 
  
  if (stopwords == TRUE){
    # pre-processing needs to be applied to stop-words as well, see https://github.com/dselivanov/text2vec/issues/228
    stopwordslist <- text_preprocessing( tm::stopwords() , settings_text_preprocess ) %>% unlist()
  }
  
  it_train = itoken(train_tokens, 
                    ids = train$doc_id,
                    # turn off progressbar because it won't look nice in rmd
                    progressbar = F)
  
  if (!is.na(test_tokens)){
    it_test = itoken(test_tokens, 
                     ids = test$doc_id,
                     # turn off progressbar because it won't look nice in rmd
                     progressbar = F)

  }
  
  
  
  if (stopwords == TRUE ){
    vocab = create_vocabulary(it = it_train,
                              ngram = ngram_range,
                              stopwords = stopwordslist )
  } else {
    vocab = create_vocabulary(it = it_train,
                              ngram = ngram_range )
  }
  
  if (!is.na(prune_vocab)){
    vocab <- vocab %>% prune_vocabulary(doc_proportion_min = prune_vocab,
                                        term_count_min = 10
                                        #doc_proportion_max = 0.5
                                        
                                        #vocab_term_max = 2000
                                        )
  }
  
  vectorizer = vocab_vectorizer(vocab)
  
  # <create document term matrices>
  dtm_train = create_dtm(it_train, vectorizer)
  if (!is.na(test_tokens)){
    dtm_test  = create_dtm(it_test, vectorizer)
  }
  #dim(dtm_train)
  #dim(dtm_test)
  
  if(tf_idf == TRUE){
    # defines tf-idf model
    # is normalized to l1 by default
    tfidf = TfIdf$new()
    # note that a new tfidf has to be specified for each train/test pair since it is a mutable object
    # so the dimensions of the tfidf has to match the dimensions from the training matrix.
    
    # fit model to train data and transform train data with fitted model
    dtm_train = fit_transform(dtm_train, tfidf)
    if (!is.na(test_tokens)){
      dtm_test = transform(dtm_test, tfidf)
    }
  }
  
  return(
    list(
      dtm_train = dtm_train,
      if (!is.na(test_tokens)){ dtm_test = dtm_test },
      vocabulary = vocab
    )
  )
  
}

# preprocessed <- preprocess(train = train, test = test, settings_preprocess)
# dtm_train <- preprocessed$dtm_train
# dtm_test <- preprocessed$dtm_test
# 
# # add dimension of dtms to settings list
# settings_preprocess[["dimension_dtm_train"]] <- dim(dtm_train)
# settings_preprocess[["dimension_dtm_test"]] <- dim(dtm_test)
