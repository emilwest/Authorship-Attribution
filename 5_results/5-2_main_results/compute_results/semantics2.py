#%%   
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri 
pandas2ri.activate()

#default_path = 'C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\'
default_path = 'C:\\Users\\Emil\\statistik\\master_thesis_stats\\'
#vocab = pd.read_csv(default_path + 'vocab.RData', sep=" ")
robjects.r['load'](default_path + 'vocab.RData')
vocab = robjects.r['allterms']

#%%   
P = np.identity(len(vocab)) # path similarities
P_wu = np.identity(len(vocab)) #  Wu-Palmer Similarity synset1.wup_similarity(synset2):

# iterate only upper diagonal since its a symmetric matrix
for i in range(0, len(vocab)):
    percent = (i+1)/(len(vocab)+1) 
    print("Row: %d out of %d" % (i +1 , len(vocab)+1) )
    print("{:.4f}".format(percent))
    for j in range(i+1, len(vocab)):
        try:
            a = wn.synsets(vocab[i])[0]
            b = wn.synsets(vocab[j])[0]
            pathsim = a.path_similarity(b)
            wusim = a.wup_similarity(b)
           
            if pathsim == None:
                #print("none")
                # adjectives etc does not have hierarchies so we set them to zero
                pathsim = 0
            if wusim == None:
                wusim = 0
            
            P[i,j] = pathsim
            P[j,i] = pathsim
            P_wu[i,j] = wusim
            P_wu[j,i] = wusim
            
        except IndexError:
            # if word is not in wordnet:
            pathsim = 0
            wusim = 0
            #lcsim = 0.01
            P[i,j] = pathsim
            P[j,i] = pathsim
            P_wu[i,j] = wusim
            P_wu[j,i] = wusim
            
#%%
#np.save(default_path + "S_pathsim", S)
np.save(default_path + "newsemm\\kaggle\\S_pathsim", P)
np.save(default_path + "newsemm\\kaggle\\S_wu", P_wu)
P_wu[10:20,10:20]
#np.savetxt(default_path + "S_pathsim", S)            
