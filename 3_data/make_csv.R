#install.packages("tidyverse")
#install.packages("tm")

library(tidyverse)
library(tm)

getwd()

# train paths 
path <- file.path("C50","C50train")
dirs <- list.dirs(path)[-1]
all_files <- list.files(path = dirs, full.names = T) 

# test paths
path_test <- file.path("C50","C50test")
dirs_test <- list.dirs(path_test)[-1]
all_files_test <- list.files(path = dirs_test, full.names = T) 


# dataframes 
df_C50train <- tibble(doc_id = seq.int(1,2500),
       path = all_files,
       fileName = gsub("C50/C50train/([a-zA-Z]+)/(.*.txt)", "\\2", path),
       author = gsub("C50/C50train/([a-zA-Z']+)/(.*.txt)", "\\1", path),
       text = ""
       ) 

df_C50test <- tibble(doc_id = seq.int(2501,5000),
                     path = all_files_test,
                     fileName = gsub("C50/C50test/([a-zA-Z]+)/(.*.txt)", "\\2", path),
                     author = gsub("C50/C50test/([a-zA-Z']+)/(.*.txt)", "\\1", path),
                     text = ""
                     ) 

# test that all authors are included:
df_C50train$author %>% unique()
df_C50test$author %>% unique()

##########################
#IMPORT TEXT AND PUT EACH TEXT IN DATAFRAME

# FOR TRAIN DATA
a <- list()
index <- 1
for(i in 1:length(dirs)){
  print(dirs[i])
  
  #author <- sub("C50/C50train/(.*)","\\1",dirs[i]) #extract author from filename 
  #print(author)
  
  # store directory name in variable
  d <- dirs[i]
  
  # create corpus that contains all documents in the directory name
  a[i] <- Corpus(DirSource(d), readerControl = list(reader=readPlain, language="en"))
  
  # loop each 50 documents in the i:th directory 
  for (j in 1:50){
    df_C50train$text[index] <- a[[i]][[j]]
    #df_C50train$author[index] <- author
    
    index <- index + 1
  }
}

# FOR TEST DATA
a2 <- list()
index <- 1
for(i in 1:length(dirs_test)){
  print(dirs_test[i])
  
  #author <- gsub("C50/C50test/(.*)","\\1",dirs_test[i]) #extract author from filename 
  #print(author)
  
  # store directory name in variable
  d <- dirs_test[i]
  
  # create corpus that contains all documents in the directory name
  a2[i] <- Corpus(DirSource(d), readerControl = list(reader=readPlain, language="en"))
  
  # loop each 50 documents in the i:th directory 
  for (j in 1:50){
    df_C50test$text[index] <- a2[[i]][[j]]
    #df_C50test$author[index] <- author
    
    index <- index + 1
  }
}


#write.csv(df_C50test, file = "df_c50test.csv")
#write.csv(df_C50train, file = "df_C50train.csv")





