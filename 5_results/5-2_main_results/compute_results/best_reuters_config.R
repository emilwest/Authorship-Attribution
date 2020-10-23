
settings_text_preprocess <- list(
  to_lower = T,
  remove_numbers = F,
  stemming = F
)
# 7e-03 for 1gram
# 2e-03 for bigram

# 0.0001 = 0.5e-04
# prune_vocab = 0.01
settings_preprocess <- list(
  stopwords = T,
  ngram_range = c(1,2), # a range (a,b). if a=b, then its a n-gram where n=a. if a!=b, then its a combination of all n-grams in the range a to b.
  prune_vocab = 7e-03, # choose a doc_proportion_min range to prune vocabulary (reduce number of features), otherwise don't prune.
  tf_idf = T      # if use tf_idf weighting,
)


kernel_settings = list(
  kernel_name = "lin"
)

settings_train = list(
  kernel = "comp",
  type = "C-svc"
)
