# wordnet similarities

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import numpy as np
import rpy2.robjects as robjects
import pandas as pd
from rpy2.robjects import pandas2ri 
pandas2ri.activate()
# synset1.path_similarity(synset2): Return a score denoting how similar two word senses are,
# based on the shortest path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. 
# The score is in the range 0 to 1.
# A score of 1 represents identity i.e. comparing a sense with itself will return 1.
# https://www.nltk.org/howto/wordnet.html

# https://linguistics.stackexchange.com/questions/9084/what-do-wordnetsimilarity-scores-mean
# https://avidml.wordpress.com/2018/12/02/semantic-similarity-approach-understand-build-and-evaluate/


# Within the path_similarity() code we see:
#if distance is None or distance < 0:
#return None
#return 1.0 / (distance + 1)

default_path = 'C:\\Users\\Emil\\statistik\\master_thesis_stats\\'
#default_path = 'C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\'
#vocab = pd.read_csv(default_path + 'vocab.RData', sep=" ")
robjects.r['load'](default_path + 'vocab.RData')
vocab = robjects.r['allterms']
vocab[4]


wn.synsets('canine')[1].hypernyms()
dog = wn.synsets('dog')[0]
cat = wn.synsets('cat')[0]
dog.path_similarity(cat)
#dog.lch_similarity(cat)
dog.wup_similarity(cat)
1/0.2

1/(4+1)

wn.synsets('hit')[0].path_similarity(wn.synsets('strike')[0])

w1 = ['dog', 'cat', 'elephant', 'squid', 'hit', '11', 'sdffd']
w2 = ['dog', 'cat', 'elephant', 'squid', 'hit', '11', 'sdffd']

w1 = ['dog', 'cat', 'elephant', 'squid', 'hit', '11', 'candy']
w2 = ['dog', 'cat', 'elephant', 'squid', 'hit', '11', 'candy']

7*7
21+21

print(w1)
vocab2 = vocab[1:10]
wn.synsets(vocab2[1])
len(w1)

wn.synsets('dog')[0]-wn.synsets('cat')[0]

#%%
S = np.identity(len(vocab2))
for i in range(0, len(vocab2)):
    for j in range(i+1, len(vocab2)):
        try:
            a = wn.synsets(vocab2[i])[0]
            b = wn.synsets(vocab2[j])[0]
            print(a,b)
            pathsim = a.path_similarity(b)
            # adjectives etc does not have hierarchies so we set them to zero
            if pathsim == None:
                print("none")
                pathsim = 0
            print(pathsim)
            S[i,j] = pathsim
            S[j,i] = pathsim
        except IndexError:
            print("err")
            pathsim=0
            print(pathsim)
            S[i,j] = pathsim
            S[j,i] = pathsim
        
        #S[i,j] = a.path_similarity(b)
        #S[i,j] = 0
        #print(i,j)
   
#%%     
        
#%%        
numcalcs = 0
S = np.identity(len(w1))
for i in range(0, len(w1)):
    for j in range(i + 1, len(w2)):
        numcalcs+=1
        print(i,j)
        try:
            a = wn.synsets(w1[i])[0]
            b = wn.synsets(w2[j])[0]
            pathsim = a.path_similarity(b)
            S[i,j] = pathsim
            S[j,i] = pathsim
        except IndexError:
            # if word is not in wordnet:
            pathsim = 0
            S[i,j] = pathsim
            S[j,i] = pathsim
#%%
           # https://arxiv.org/ftp/arxiv/papers/1310/1310.8059.pdf  # Semantic similarity Measures Approaches
#%%        
S = np.identity(len(vocab)) # path similarities
S_wu = np.identity(len(vocab)) #  Wu-Palmer Similarity synset1.wup_similarity(synset2):

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
                pathsim = 0.01
            if wusim == None:
                wusim = 0.01
            
            S[i,j] = pathsim
            S[j,i] = pathsim
            S_wu[i,j] = wusim
            S_wu[j,i] = wusim
            
        except IndexError:
            # if word is not in wordnet:
            pathsim = 0.01
            wusim = 0.01
            #lcsim = 0.01
            S[i,j] = pathsim
            S[j,i] = pathsim
            S_wu[i,j] = wusim
            S_wu[j,i] = wusim
            
#%%
#np.save(default_path + "S_pathsim", S)
np.save(default_path + "sem\\reuters\\S_pathsim", S)
np.save(default_path + "sem\\reuters\\S_wu", S_wu)
#np.savetxt(default_path + "S_pathsim", S)            
