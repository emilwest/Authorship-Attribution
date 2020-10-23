library(tidyverse)
#install.packages("jsonlite")
library(jsonlite)
library(tm)

getwd()
path <- file.path("csv",
                  "private_datasets",
                  "judgment",
                  "relevant_folders"
)
path
dirs <- list.dirs(path)[-1]
all_files <- list.files(path = dirs, full.names = T) 
which(all_files)
seq(all_files)

df_judgment <-  tibble(doc_id = seq(all_files),
                       path = all_files,
                       fileName = gsub(".*/([a-zA-Z]+[0-9]+).txt", "\\1", path),
                       author = gsub(".*/([a-zA-Z]+)[0-9]+.txt", "\\1", path),
                       text = "",
                       train = 1) 

a <- list()
index <- 1
for(i in 1:length(dirs)){
  print(dirs[i])
  # store directory name in variable
  d <- dirs[i]
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d, encoding = "UTF-8"), readerControl = list(reader=readPlain, language="en"))
  num_files_in_dir <- list.files(d) %>% length()
  # loop documents in the i:th directory 
  for (j in 1:num_files_in_dir){
    df_judgment$text[index] <- a[[i]][[j]]
    index <- index + 1
  }
}

write_csv(df_judgment, path = "csv/private_datasets/judgment.csv")
