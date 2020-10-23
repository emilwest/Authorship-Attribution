


# http://text2vec.org/vectorization.html#basic_transformations

getwd()

df <- read_csv("csv/reuters/c50.csv",
               col_types = cols(
                 X1 = col_double(),
                 doc_id = col_double(),
                 path = col_character(),
                 fileName = col_character(),
                 author = col_factor(),
                 text = col_character(),
                 train = col_double()
               )
)
df <- df %>% select(-X1)
df

y.train <- df %>% filter(train==1) %>% select(author) %>% as_vector()
y.test <- df %>% filter(train==0) %>% select(author) %>% as_vector()


train <- df %>% filter(train==1) 
test <- df %>% filter(train==0)

#y.train
write.table(y.train, file = "y.train")
write.table(y.test, file = "y.test")
#as.factor(y.train)
