library(tidyverse)
#install.packages("jsonlite")
library(jsonlite)
library(tm)

getwd()
path <- file.path("csv",
                  "private_datasets",
                  "C10",
                  "C10train"
                  )
dirs <- list.dirs(path)[-1]
all_files <- list.files(path = dirs, full.names = T) 

# test paths
path_test <- file.path("csv",
                       "private_datasets",
                       "C10",
                       "C10test"
)
dirs_test <- list.dirs(path_test)[-1]
all_files_test <- list.files(path = dirs_test, full.names = T) 


# dataframes 
df_C10train <- tibble(doc_id = seq.int(1,500),
                      path = all_files,
                      fileName = gsub(".*/C10/C10train/([a-zA-Z]+)/(.*.txt)", "\\2", path),
                      author = gsub(".*/C10/C10train/([a-zA-Z']+)/(.*.txt)", "\\1", path),
                      text = "",
                      train = 1
) 

df_C10test <- tibble(doc_id = seq.int(501,1000),
                     path = all_files_test,
                     fileName = gsub(".*/C10/C10test/([a-zA-Z]+)/(.*.txt)", "\\2", path),
                     author = gsub(".*/C10/C10test/([a-zA-Z']+)/(.*.txt)", "\\1", path),
                     text = "",
                     train = 0
) 



# FOR TRAIN DATA
a <- list()
index <- 1
for(i in 1:length(dirs)){
  print(dirs[i])
  d <- dirs[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  # loop each 50 documents in the i:th directory 
  for (j in 1:50){
    df_C10train$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}

# FOR TEST DATA
a2 <- list()
index <- 1
for(i in 1:length(dirs_test)){
  print(dirs_test[i])
  d <- dirs_test[i]
  # create corpus that contains all documents in the directory name
  a2[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  # loop each 50 documents in the i:th directory 
  for (j in 1:50){
    df_C10test$text[index] <- a2[[i]][[j]]
    index <- index + 1
  }
}
df_C10train$text[1]
df_C10test$text[1]


df_C10all <- rbind(df_C10train,df_C10test)
write_csv(df_C10all, path = "csv/private_datasets/C10.csv")

