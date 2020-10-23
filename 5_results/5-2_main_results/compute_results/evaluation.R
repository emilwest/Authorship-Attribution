
metrics <- function(predictions, y){
  t <- table(predicted=predictions, actual=y)
  n <- sum(t)
  accuracy <- sum(diag(t))/n %>% round(4)
  
  precision <- diag(t)/colSums(t)
  precision[is.na(precision)==TRUE] = 0 # convert Na/NaN to 0
  
  recall <- diag(t)/rowSums(t)
  f1 <- 2* (precision*recall)/(precision + recall)
  f1[is.na(f1)==TRUE] = 0
  
  # the arithmetic means
  f1_macro <- mean(f1) %>% round(4) # sum(f1)/50
  precision_macro <- mean(precision) %>% round(4)
  recall_macro <- mean(recall) %>% round(4)
  
  return(
  list(
    conf_matrix = t,
    accuracy = accuracy,
    precision_macro = precision_macro,
    recall_macro = recall_macro,
    f1_macro = f1_macro,
    precision,
    recall,
    f1
  )
    )
}




# precision tp / (tp + fp)
# recall tp / (tp + fn)
metric_on_confmat <- function(confmat){
  
  n <- sum(confmat)
  accuracy <- sum(diag(confmat))/n %>% round(4)

  precision <- diag(confmat)/colSums(confmat)
  precision[is.na(precision)==TRUE] = 0 # convert Na/NaN to 0
    
  recall <- diag(confmat)/rowSums(confmat)
  
  
  
  f1 <- 2* (precision*recall)/(precision + recall)
  f1[is.na(f1)==TRUE] = 0
  
  # the arithmetic means
  f1_macro <- mean(f1) %>% round(4) # sum(f1)/50
  precision_macro <- mean(precision) %>% round(4)
  recall_macro <- mean(recall) %>% round(4)
  
  metricstable = table(precisions = precision, recalls = recall, f1 = f1)
  
  return(
    list(
      accuracy = accuracy,
      precision_macro = precision_macro,
      recall_macro = recall_macro,
      f1_macro = f1_macro,
      precision = precision,
      recall = recall,
      f1 = f1
    )
  )
  
}



###
# experiments
# https://rdrr.io/cran/caret/man/recall.html
# precision: 
# 
# results$conf_matrix["HeatherScoffield",] # has zero predictions!!!!!!!!!!!!!!!!!!!!!
# 
# rowSums(results$conf_matrix)
# colSums(results$conf_matrix) # 50
# diags = diag(results$conf_matrix) 
# precision <- diags/rowSums(results$conf_matrix)
# precision[is.na(precision)==TRUE] = 0 # convert Na/NaN to 0
# 
# recall <- diags/colSums(results$conf_matrix)
# results$conf_matrix
# f1 <- 2* (precision*recall)/(precision + recall)
# f1[is.na(f1)==TRUE] = 0
# 
# f1_macro <- mean(f1) # sum(f1)/50
# precision_macro <- mean(precision)
# recall_macro <- mean(recall)
# 
