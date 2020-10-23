library(tidyverse)

df <- read_csv("csv/private_datasets/pan18_problem1.csv",
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
y.test <- df %>% filter(train==0) %>% select(author) %>% as_vector()

train <- df %>% filter(train==1) 
test <- df %>% filter(train==0)

write.table(y.train, file = "y.train")
write.table(y.test, file = "y.test")