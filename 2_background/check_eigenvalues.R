
# https://stackoverflow.com/questions/29644180/gram-matrix-kernel-in-svms-not-positive-semi-definite
D <- as.matrix(dtm_train)
G <- D%*%t(D) + 1e-10*diag(nrow(dtm_train)) # add small constant to account for numerical rounding errors
eigenvals <- eigen(G)

Dkaggle <- as.matrix(dtm_train)
Gkaggle <- Dkaggle%*%t(Dkaggle) + 1e-10*diag(nrow(dtm_train)) # add small constant to account for numerical rounding errors
eigenvalskaggle <- eigen(Gkaggle)

eigenvalskaggle

eigenvals$values %>% 
  plot()

eigenvals$values %>% 
  as_tibble() %>%
  ggplot(aes(y=value,x=1:2500)) + geom_point(size=0.5) + theme_light() +labs(title = "Kernel matrix eigenvalues",
                                                                     subtitle = "Reuters data. Dimension 2500x2500",
                                                                     x = "Index" )


# plot the kernel matrix
#library(reshape2)
#library(ggplot2)
#m = matrix(rnorm(20),5)
#ggplot(melt(m), aes(Var1,Var2, fill=value)) + geom_raster()


any(eigenvals$values<0) 
all(eigenvals$values>=0) #TRUE

