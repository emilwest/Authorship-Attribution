library(tidyverse)

df <- read_tsv("csv/private_datasets/imdb62/imdb62.txt", 
         quote = "", # disables quotation 
         col_names = FALSE, # original file does not have colnames included
         col_types = cols(
           X1 = col_double(),
           X2 = col_factor(),
           X3 = col_double(),
           X4 = col_double(),
           X5 = col_character(),
           X6 = col_character()
         ))

# original column names:
colnames(df) <-  c("reviewId",	"userId",	"itemId",	"rating",	"title",	"content")
df

# new column names:
colnames(df) <-  c("doc_id",	"author",	"itemId",	"rating",	"title",	"text")
df

# example where quotation occurs, which is \"
df %>% filter(itemId == 1222814) %>% select(text) %>% as.character()

# this dataset is not pre-divided into train/test-sets so we use it all for cross-validation or split it
y.train <- df %>% select(author) %>% as_vector()
str(y.train)
train <- df 

write.table(y.train, file = "y.train")
