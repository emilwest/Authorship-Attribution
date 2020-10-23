library(tidyverse)
#install.packages("jsonlite")
library(jsonlite)
library(tm)

#csv\private_datasets\pan12-authorship-attribution-corpora\pan12-authorship-attribution-dataset-2015-10-20\pan12-authorship-attribution-test-dataset-problem-a-2015-10-20
getwd()

# train paths 
path <- file.path("csv",
                  "private_datasets",
                  "pan12-authorship-attribution-corpora",
                  "pan12-authorship-attribution-dataset-2015-10-20",
                  "pan12-authorship-attribution-test-dataset-problem-a-2015-10-20" )


dirs <- list.dirs(path)[-1]
dirs_train <- dirs[-4]
dirs_test <- dirs[4]
all_files <- list.files(path = dirs, full.names = T) 
# contains true labels for test set documents
ground_truth <- jsonlite::fromJSON( paste0(path,"/ground-truth.json"))[[1]] 


train_ind <- grep("/known[0-9]+.txt", all_files) #training files
test_ind <- grep("/unknown[0-9]+.txt", all_files)
length(train_ind)

# dataframes 
df_pan12 <- tibble(doc_id = train_ind ,
                      path = all_files[train_ind],
                      fileName = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\2", path),
                      author = gsub(".*/(candidate[0-9]+)/(known[0-9]+).txt", "\\1", path),
                      text = "",
                      train = 1) 

df_pan12_test <-  tibble(doc_id = test_ind ,
                         path = all_files[test_ind],
                         fileName = gsub(".*/(unknown[0-9]+).txt", "\\1", path),
                         author = ground_truth$`true-author`,
                         text = "",
                         train = 0) 

df_pan12
gsub(".*/(candidate[0-9]+)/(known[0-9]+.txt)", "\\2", all_files[grep("/known[0-9]+.txt", all_files)] )

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
    df_pan12$text[index] <- a[[i]][[j]]
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
    df_pan12_test$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}


df_pan12_all <- rbind(df_pan12,df_pan12_test)
write_csv(x = df_pan12_all, path = "csv/private_datasets/pan12_a.csv")
