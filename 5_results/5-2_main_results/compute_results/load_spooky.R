# load spooky data set

#id - a unique identifier for each sentence
#text - some text written by one of the authors
#author - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# Edgar Allan Poe, HP Lovecraft and Mary Shelley.

df <- read_csv("csv/spooky/train.csv", col_types = cols(
  id = col_character(),
  text = col_character(),
  author = col_factor()
))
# change name to doc_id for consistency
df$doc_id <- df$id
df <- df %>% select(-id)
df
# the test data does not include actual labels so we cannot consider it
# test <- read_csv("csv/spooky/test/test.csv", col_types = cols(
#   id = col_character(),
#   text = col_character()
# ))
#train

#y.train <- train %>% select(author) %>% as_vector()
#y.test <- df %>% filter(train==0) %>% select(author) %>% as_vector()


smp_size <- floor(0.7*nrow(df))
set.seed(42)
train_ind <- sample(seq_len(nrow(df)), size = smp_size , replace = F)
train <- df[train_ind, ]
test <- df[-train_ind, ]
train
test
y.train <- train %>% select(author) %>% as_vector()
y.test <- test %>% select(author) %>% as_vector()

write.table(y.train, file = "y.train")
write.table(y.test, file = "y.test")
