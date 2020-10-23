# creating tables and figures for results
library(tidyverse)

# if (dataset == "reuters"){
#   all_results_dirs <- list.dirs("latest_results/reuters/", full.names = T)[-1]
# }
# if (dataset == "spooky"){
#   all_results_dirs <- list.dirs("latest_results/spooky/", full.names = T)[-1]
# }

all_results_dirs <- list.dirs("latest_results/reuters/", full.names = T)[-1]



# contains_stop_setting <- grep("stop", all_results_dirs)
# 22 %in% contains_stop_setting
# settings <- paste0(all_results_dirs[21], "//all_settings.csv")
# dummy <- read_csv(settings)
# colnames(dummy) <- c("variable","value")
# dummy
# 
# if (21 %in% contains_stop_setting ==F){
#   dummy <- dummy %>% add_row(variable = "stop", value = "TRUE", .before = 1)
# }


dummy <- read_csv(paste0(all_results_dirs[1], "//all_settings.csv")) %>% t()
results_df <- matrix(NA, nrow=0, ncol=ncol(dummy)) %>% as.data.frame()
colnames(results_df) <- dummy[1,]
results_df

# store all results in single data fram by iterating through the folders in latest_results/
for (i in grep("ngram__", all_results_dirs ) ){
  #print(i)
  #print(all_results_dirs[i])
  settings <- paste0(all_results_dirs[i], "//all_settings.csv")
  dummy <- read_csv(settings) 
  #colnames(dummy) <- c("variable","value")
  #if (i %in% contains_stop_setting ==F){
  #  dummy <- dummy %>% add_row(variable = "stop", value = "TRUE", .before = 1)
  #}
  dummy <- t(dummy)
  results_df[i, ] <- dummy[2,]  # results_df[i, ] works only for data.frame and not tibbles
}

# now convert to tibble and parse columns with type_convert
results_df <- results_df %>% as_tibble() %>% type_convert()
results_df


# by ngrams
results_df %>% filter(ngram_range1==1, ngram_range2==1)
results_df %>% filter(ngram_range1==1, ngram_range2==2)
results_df %>% filter(ngram_range1==2, ngram_range2==2)
results_df %>% filter(ngram_range1==3, ngram_range2==3)

# THE BEST MODEL
# model with highest accuracy accuracy
a <- results_df %>% arrange(  desc(accuracy) ) %>% head(1) %>% select(accuracy, f1_macro, precision_macro, recall_macro, everything() ) 

# model with highest f1
results_df %>% arrange(  desc(f1_macro) ) %>% head(1) %>% select(f1_macro, everything())

#model with highest precision 
results_df %>% arrange(  desc(precision_macro) ) %>% head(1) %>% select(precision_macro, everything())

# model with highest recall
results_df %>% arrange(  desc(recall_macro) ) %>% head(1) %>% select(recall_macro, everything())
library(reshape2)

# accuracy equals recall
p1<- results_df %>% 
  filter(ngram_range1==1,
         ngram_range2==1,
         kernel_name=="linear",
         tf_idf==TRUE) %>%
  arrange( dimension_dtm_train2 ) %>%
  select(dimension_dtm_train2, f1_macro, precision_macro, recall_macro) %>% 
  melt(id.vars="dimension_dtm_train2") %>% 
  ggplot(aes(x=dimension_dtm_train2, y=value , color=variable, group=variable) ) + 
  geom_point() + 
  geom_line() +
  labs(
       subtitle = "1-gram, linear kernel",
       x = "Features"
       ) +  theme(legend.title = element_blank(),
                  axis.title.y = element_blank(),
                  axis.title.x = element_blank() ) 
p1

p2 <- results_df %>% 
  filter(ngram_range1==2,
         ngram_range2==2,
         kernel_name=="linear",
         tf_idf==TRUE) %>%
  arrange( dimension_dtm_train2 ) %>%
  select(dimension_dtm_train2, f1_macro, precision_macro, recall_macro) %>% 
  melt(id.vars="dimension_dtm_train2") %>% 
  ggplot(aes(x=dimension_dtm_train2, y=value , color=variable, group=variable) ) + 
  geom_point() + 
  geom_line()  +
  labs(
       subtitle = "2-gram, linear kernel") + 
  theme(legend.title = element_blank(),
                        axis.title.y = element_blank(),
                        axis.title.x = element_blank() )
p2


p3 <- results_df %>% 
  filter(ngram_range1==3,
         ngram_range2==3,
         kernel_name=="linear",
         tf_idf==TRUE) %>%
  arrange( dimension_dtm_train2 ) %>%
  select(dimension_dtm_train2, f1_macro, precision_macro, recall_macro) %>% 
  melt(id.vars="dimension_dtm_train2") %>% 
  ggplot(aes(x=dimension_dtm_train2, y=value , color=variable, group=variable) ) + 
  geom_point() + 
  geom_line()  +
  labs(
       subtitle = "3-gram, linear kernel",
       x = "Number of features"
       ) +  theme(legend.title = element_blank(),
                        axis.title.y = element_blank() )
p3

#install.packages("gridExtra")
library(gridExtra)
grid.arrange(p1,p2,p3)
install.packages("kableExtra")
library(kableExtra)
save(results_df, file = "results_df_reuterslatest.RData")

results_df %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name) %>%
  kable(format = "latex",
        caption = "Performance for different configurations for the Kaggle and Reuters data sets", 
        label = "tab:semlinear",
        col.names = c("remove stopwords" , "ngram ", "prune" ,
                      "tf-idf" , "dimensiondtm" , "tolower",
                       "remove numbers" , "stemming" , "kernel" ,
                      "accuracy" , "f1 macro" , "precision macro" , "recall macro" , "C"),
        booktabs=T) %>%
  kable_styling(latex_options = "striped") %>%
  pack_rows("Reuters", 1, 3, latex_gap_space = "0.5em") 
  #pack_rows("Kaggle", 4, 5, latex_gap_space = "0.5em") 

  

