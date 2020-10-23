library(tidyverse)

df <- read_csv("csv/private_datasets/judgment.csv",
               col_types = cols(
                 doc_id = col_double(),
                 path = col_character(),
                 fileName = col_character(),
                 author = col_factor(),
                 text = col_character(),
                 train = col_double()
               )
               )
df

y.train <- df %>% filter(train==1) %>% select(author) %>% as_vector()
str(y.train)
train <- df %>% filter(train==1) 

write.table(y.train, file = "y.train")
