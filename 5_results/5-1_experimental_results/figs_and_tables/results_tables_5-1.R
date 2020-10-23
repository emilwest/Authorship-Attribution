load("results_df_reuters.RData")
load("results_df_spooky.RData")
all_results <- bind_rows(results_df_reuters, results_df_spooky)

library(tidyverse)


all_results %>% filter(ngram_range1==1, ngram_range2==1)
all_results %>% filter(ngram_range1==2, ngram_range2==2)
all_results %>% filter(ngram_range1==3, ngram_range2==3)

# THE BEST MODEL
# model with highest accuracy accuracy
# best overall
a <- all_results %>% filter(dataset=="spooky") %>% arrange(  desc(accuracy) ) %>% head(1) %>% select(accuracy, f1_macro, precision_macro, recall_macro, dataset, everything() ) 
b <- all_results %>% filter(dataset=="reuters") %>% arrange(  desc(accuracy) ) %>% head(1) %>% select(accuracy, f1_macro, precision_macro, recall_macro, dataset, everything() ) 



# model with highest f1
all_results %>% filter(dataset=="spooky")  %>% arrange(  desc(f1_macro) ) %>% head(1) %>% select(f1_macro, everything())
all_results %>% filter(dataset=="reuters")  %>% arrange(  desc(f1_macro) ) %>% head(1) %>% select(f1_macro, everything())

#model with highest precision 
all_results %>% filter(dataset=="spooky") %>% arrange(  desc(precision_macro) ) %>% head(1) %>% select(precision_macro, everything())
all_results %>% filter(dataset=="reuters") %>% arrange(  desc(precision_macro) ) %>% head(1) %>% select(precision_macro, everything())

# model with highest recall
all_results %>% filter(dataset=="spooky")  %>% arrange(  desc(recall_macro) ) %>% head(1) %>% select(recall_macro, everything())
all_results %>% filter(dataset=="reuters")  %>% arrange(  desc(recall_macro) ) %>% head(1) %>% select(recall_macro, everything())

library(reshape2)

ss <- c(`TRUE` = "Stopwords", `FALSE` = "No stopwords")
dd <- c(`reuters` = "Dataset: Reuters", `spooky` = "Dataset: Kaggle" )
nn <- c(`1` = "1-gram", `2` = "2-gram",`3` = "3-gram")
#tf <- c(`TRUE` = "Tf-idf: True", `FALSE` = "Tf-idf: False")

all_results %>% 
  filter(kernel_name=="linear",
         tf_idf == T) %>%
  arrange( dimension_dtm_train2 ) %>%
  select(dimension_dtm_train2, f1_macro, precision_macro, recall_macro, stopwords, dataset, ngram_range1) %>% 
  melt(id.vars=c("dimension_dtm_train2","stopwords","dataset", "ngram_range1")) %>% 
  ggplot(aes(x=dimension_dtm_train2, y=value , color=variable, group=variable) ) + 
  geom_point() + 
  geom_line() +
  facet_grid( dataset  ~ stopwords + ngram_range1 , 
              scales = "free", 
              labeller = labeller(stopwords = ss , dataset = dd, ngram_range1 = nn)
              ) +
  labs(
    title = "The effect on classification perfomance for different configurations",
    x = "Number of features"
  ) +  theme(legend.title = element_blank(),
             axis.title.y = element_blank())


all_results %>% filter(tf_idf==F, dataset=="spooky") %>% select(tf_idf, everything() )
376+376



p2 <- all_results %>% 
  filter(kernel_name=="linear",
         tf_idf == T) %>%
  arrange( dimension_dtm_train2 ) %>%
  select(dimension_dtm_train2, f1_macro, precision_macro, recall_macro, stopwords, dataset, ngram_range1) %>% 
  melt(id.vars=c("dimension_dtm_train2","stopwords","dataset", "ngram_range1")) %>% 
  ggplot(aes(x=dimension_dtm_train2, y=value , color=variable, group=variable) ) + 
  geom_point() + 
  geom_line() +
  facet_wrap( dataset  ~ stopwords + ngram_range1 , 
              scales = "free", 
              labeller = labeller(stopwords = ss , dataset = dd, ngram_range1 = nn),
              ncol=3) +
  labs(
    title = "The effect on classification perfomance for different configurations",
    x = "Number of features"
  ) +  theme(legend.title = element_blank(),
             axis.title.y = element_blank()
              )



#install.packages("kableExtra")
library(kableExtra)
results_df_reuters <- results_df_reuters %>% select(-X1)

n1 <- results_df_reuters %>% filter(ngram_range1 == 1 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)

n2 <- results_df_reuters %>% filter(ngram_range1 == 2 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)
n2

n3 <- results_df_reuters %>% filter(ngram_range1 == 3 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)
n3
reut <- rbind(n1,n2,n3) 

rbind(n1,n2,n3) %>% 
  kable(format = "latex", booktabs=T)





results_df_spooky <- results_df_spooky %>% select(-X1)

n1 <- results_df_spooky %>% filter(ngram_range1 == 1 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)

n2 <- results_df_spooky %>% filter(ngram_range1 == 2 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)
n2

n3 <- results_df_spooky %>% filter(ngram_range1 == 3 ) %>% 
  select(-dimension_dtm_train1, -dimension_dtm_test1, 
         -ngram_range2, -dimension_dtm_test2, -type, -kernel_name, -kernel, -dataset) %>%
  arrange( desc(f1_macro) ) %>% head(5)
n3
kag <- rbind(n1,n2,n3)

rbind(kag,reut) %>% 
  kable(format = "latex",
                caption = "Performance for different configurations for the Kaggle and Reuters data sets", label = "tab:configs",
        booktabs=T) %>%
  kable_styling(latex_options = "striped") %>%
  pack_rows("Kaggle", 1, 15, latex_gap_space = "0.5em") %>% 
  pack_rows("Reuters", 16, 30, latex_gap_space = "0.5em") 

?pack_rows
# rbind(n1,n2,n3) %>% 
#   kable(format = "latex", booktabs=T)







