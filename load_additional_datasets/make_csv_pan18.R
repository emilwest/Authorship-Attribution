library(tidyverse)
#install.packages("jsonlite")
library(jsonlite)
library(tm)

getwd()
path <- file.path("csv",
                  "private_datasets",
                  "pan18-cross-domain-authorship-attribution-dataset",
                  "pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02",
                  "problem00001")


dirs <- list.dirs(path)[-1]
length(dirs)
dirs_train <- dirs[-length(dirs)]
dirs_test <- dirs[length(dirs)] # the last directory is the test folder
all_files <- list.files(path = dirs, full.names = T) 
# contains true labels for test set documents
ground_truth <- jsonlite::fromJSON( paste0(path,"/ground-truth.json"))[[1]] 

train_ind <- grep("/known[0-9]+.txt", all_files) #training files
test_ind <- grep("/unknown[0-9]+.txt", all_files)

df_pan18 <- tibble(doc_id = train_ind  ,
                   path = all_files[train_ind],
                   fileName = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\2", path),
                   author = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\1", path),
                   text = "",
                   train = 1) 

df_pan18_test <-  tibble(doc_id = test_ind ,
                         path = all_files[test_ind],
                         fileName = gsub(".*/(unknown[0-9]+).txt", "\\1", path),
                         author = ground_truth$`true-author`,
                         text = "",
                         train = 0) 
a <- list()
index <- 1
for(i in 1:length(dirs_train)){
  print(dirs_train[i])
  # store directory name in variable
  d <- dirs_train[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  num_files_in_dir <- list.files(d) %>% length()
  # loop documents in the i:th directory 
  for (j in 1:num_files_in_dir){
    df_pan18$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}
a <- list()
index <- 1
for(i in 1:length(dirs_test)){
  print(dirs_test[i])
  # store directory name in variable
  d <- dirs_test[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  num_files_in_dir <- list.files(d) %>% length()
  # loop documents in the i:th directory 
  for (j in 1:num_files_in_dir){
    df_pan18_test$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}
df_pan18$text[1]
df_pan18_test$text[1]

df_pan18_all <- rbind(df_pan18,df_pan18_test)
write_csv(x = df_pan18_all, path = "csv/private_datasets/pan18_problem1.csv")



###############################


getwd()
path <- file.path("csv",
                  "private_datasets",
                  "pan18-cross-domain-authorship-attribution-dataset",
                  "pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02",
                  "problem00002")


dirs <- list.dirs(path)[-1]
length(dirs)
dirs_train <- dirs[-length(dirs)]
dirs_test <- dirs[length(dirs)] # the last directory is the test folder
all_files <- list.files(path = dirs, full.names = T) 
# contains true labels for test set documents
ground_truth <- jsonlite::fromJSON( paste0(path,"/ground-truth.json"))[[1]] 

train_ind <- grep("/known[0-9]+.txt", all_files) #training files
test_ind <- grep("/unknown[0-9]+.txt", all_files)

df_pan18 <- tibble(doc_id = train_ind  ,
                   path = all_files[train_ind],
                   fileName = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\2", path),
                   author = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\1", path),
                   text = "",
                   train = 1) 

df_pan18_test <-  tibble(doc_id = test_ind ,
                         path = all_files[test_ind],
                         fileName = gsub(".*/(unknown[0-9]+).txt", "\\1", path),
                         author = ground_truth$`true-author`,
                         text = "",
                         train = 0) 
a <- list()
index <- 1
for(i in 1:length(dirs_train)){
  print(dirs_train[i])
  # store directory name in variable
  d <- dirs_train[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  num_files_in_dir <- list.files(d) %>% length()
  # loop documents in the i:th directory 
  for (j in 1:num_files_in_dir){
    df_pan18$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}
a <- list()
index <- 1
for(i in 1:length(dirs_test)){
  print(dirs_test[i])
  # store directory name in variable
  d <- dirs_test[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  num_files_in_dir <- list.files(d) %>% length()
  # loop documents in the i:th directory 
  for (j in 1:num_files_in_dir){
    df_pan18_test$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}
df_pan18$text[1]
df_pan18_test$text[1]

df_pan18_all <- rbind(df_pan18,df_pan18_test)
write_csv(x = df_pan18_all, path = "csv/private_datasets/pan18_problem2.csv")
