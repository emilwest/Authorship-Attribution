# %%

from sklearn import svm
from sklearn import metrics
import pandas as pd # for reading file
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import rpy2.robjects as robjects
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# FOR ROC CURVE:
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



default_path = 'C:\\Users\\Emil\\statistik\\master_thesis_stats\\'
#default_path = 'C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\'


robjects.r['load'](default_path + 'dtm_train.RData')
#robjects.r['load']("C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_train.RData")
dtm_train = robjects.r['dtm_train']
dtm_train = robjects.r['as.matrix'](dtm_train)
dtm_train = np.array(dtm_train)


#robjects.r['load'](default_path + 'dtm_test.RData')
#robjects.r['load']("C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\dtm_test.RData")
#dtm_test = robjects.r['dtm_test']
#dtm_test = robjects.r['as.matrix'](dtm_test)
#dtm_test = np.array(dtm_test)

# robjects.r['load'](default_path + 'Rmatrix.RData')
# R = robjects.r['R']
# R = robjects.r['as.matrix'](R)
# R = np.array(R) # diagonal idf matrix

ytrain = pd.read_csv(default_path + 'y.train', sep=" ")
#ytrain = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\y.train', sep=" ")
ytrain = ytrain['x'].astype('category')

#ytest = pd.read_csv(default_path + 'y.test', sep=" ")
#ytest = pd.read_csv('C:\\Users\\emwe9516\\Desktop\\master_thesis_stats-master\\y.test', sep=" ")
#ytest = ytest['x'].astype('category')
type(dtm_train)
# %%


############################################
# using kernel matrix
# %%
start = timer()
gram = np.dot(dtm_train, dtm_train.T)
end = timer()
print(end - start) 
kernel_test = dtm_test@dtm_train.T # dot product

type(gram)

# %%

#%%
#from sklearn.preprocessing import KernelCenterer


P = np.load(default_path + "newsemm\\kaggle\\S_pathsim.npy", allow_pickle=True)
P_wu = np.load(default_path + "newsemm\\kaggle\\S_wu.npy", allow_pickle=True)
S = P
S = P_wu
# S = R@P
# S_wu = R@P_wu
S = P
S_wu = P_wu

S1 = np.where(S==0,0.001, S)
S = S1

A = dtm_train@S
B = S.T@dtm_train.T
G = A@B
kernel_test = dtm_test@S@S.T@dtm_train.T 

#transformer = KernelCenterer().fit(G)
#G = transformer.transform(G)
#kernel_test = transformer.transform(kernel_test)


G = dtm_train@dtm_train.T
G
kernel_test = dtm_test@dtm_train.T 



fig = plt.figure()
fig.set_size_inches(11,11, forward=True)
plt.imshow(G, interpolation='nearest')

plt.savefig(default_path + "kernelviz.pdf", dpi=100)

kernel_test = dtm_test@dtm_train.T 
kernel_test = dtm_test@S@S.T@dtm_train.T 
kernel_test = dtm_test@S_wu@S_wu.T@dtm_train.T 

#%%



####################################################
#tuned_parameters = [  {'kernel': ['linear'], 'C': [ 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3, 2e4]} ]
#tuned_parameters = [  {'kernel': ['linear'], 'C': [ 2e1, 2e2, 2e3, 2e4]} ]
tuned_parameters = [  {'kernel': ['precomputed'], 'C': [ 100, 200, 400, 800, 1000, 2000] } ]
tuned_parameters = [  {'kernel': ['precomputed'], 'C': [ 1,2,4,10,20,100] } ]
tuned_parameters = [  {'kernel': ['precomputed'], 'C': [ 0.02, 0.2, 1] } ]
tuned_parameters = [  {'kernel': ['precomputed'], 'C': [ 0.02, .002,.0002,.00002, .001,.005] } ]
tuned_parameters = [  {'kernel': ['precomputed'], 'C': [ 0.5, .6, .7, .8, .9, 1] } ]
#
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
#
##scores = ['precision', 'recall']
#scores = ['precision']
#
#
# %%
start = timer()
##for score in scores:
##print("# Tuning hyper-parameters for %s" % score)
##print()
#
clf = GridSearchCV(
    svm.SVC(decision_function_shape='ovo', kernel='precomputed', cache_size=2000), 
    tuned_parameters, 
    scoring='precision_macro',
    n_jobs = 1,
    verbose = 10,
    cv = 5
    
)
clf.fit(G, ytrain)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = ytest, clf.predict(kernel_test)
print(classification_report(y_true, y_pred))
print()

end = timer()
print(end - start) 
clf.best_params_['C']
bestC = clf.best_params_['C']
#bestC = 0.005

##################################################################################
# CROSS VALIDAION WITHOUT GRIDSEARCH

# %%
bestC = 1
bestC = 10
bestC = 20
bestC = 100

clf = svm.SVC(decision_function_shape='ovr', kernel='precomputed', C=bestC, cache_size=1600, probability=True)
#clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=bestC, cache_size=1600)
scores = cross_val_score(clf, G, ytrain, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#bestC = clf.best_params_['C']

# %%

# %%
# https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
ytrain_bin = label_binarize(ytrain, classes = ytrain.cat.categories)
ytest_bin = label_binarize(ytest, classes = ytest.cat.categories)
n_classes = ytest_bin.shape[1]

clfprecom_prob = svm.SVC( kernel='precomputed', decision_function_shape='ovr',
                    C=bestC, cache_size=1600,
                    probability = True)
clfprecom_prob.fit(G, ytrain)
preds_prob1 = clfprecom_prob.predict_proba(kernel_test)
preds_prob2 = clfprecom_prob.decision_function(kernel_test)
preds_prob3 = clfprecom_prob.predict(kernel_test)
preds_prob3 = pd.Series(preds_prob3)
metrics.accuracy_score(ytest,preds_prob3) # 0.7899216887980933
preds = preds_prob3

#decisions = (preds_prob1 >= 0.1).astype(int)



#from sklearn.preprocessing import binarize
#
#bins = binarize(preds_prob1, threshold=0.3)
#confusion_matrix(ytest_bin, bins)


#from sklearn.preprocessing import LabelBinarizer
#lb = LabelBinarizer()
#lb.fit(preds_prob3)
#a = lb.transform(preds_prob3)
#orig = lb.inverse_transform(a)
#orig


# why negative values from the decision function? answer:
# https://stackoverflow.com/questions/46820154/negative-decision-function-values
# You need to consider the decision_function and the prediction separately. 
# The decision is the distance from the hyperplane to your sample. 
# This means by looking at the sign you can tell if your sample is located right or left to the hyperplane. 
# So negative values are perfectly fine and indicate the negative class ("the other side of the hyperplane").


#clfprecom = svm.SVC(decision_function_shape='ovo', kernel='linear', C=bestC, cache_size=2000)
#clfprecom.fit(dtm_train, ytrain)

# https://stackoverflow.com/questions/15015710/how-can-i-know-probability-of-class-predicted-by-predict-function-in-support-v

# setting threshold: 
# y_pred = (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool) # set threshold as 0.3


#prob_per_class_dictionary = dict(zip(clfprecom.classes_, preds))

# TO DO : SAVE PROBABILITIES 

sum(preds)
preds = pd.Series(preds)

metrics.accuracy_score( ytest , preds)  

confmat = confusion_matrix(ytest, preds)
report = classification_report(ytest, preds, target_names=ytest.cat.categories,  output_dict=True)
print(classification_report(ytest, preds, target_names=ytest.cat.categories) )

report_df =  pd.DataFrame(report).transpose()
bestC = {'bestC' :  [bestC]}
pd.DataFrame.from_dict(bestC)
print(bestC)

report_df.to_csv(default_path + 'report.csv')
np.savetxt(default_path + 'confmat.csv', np.asarray(confmat), delimiter=",", fmt='%1i')
pd.DataFrame.from_dict(bestC).to_csv(default_path + 'bestC.csv')
np.savetxt(default_path + 'preds_prob.txt', preds_prob1, delimiter=" ")

# %%
41/50
24/50

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(ytest,
                                    preds,
                                    normalize=False,
                                    figsize=(8,8),
                                    hide_counts=False,
                                    x_tick_rotation=-90
                                    )
plt.savefig(default_path + "\\fig\\confmat_kaggle_lin.pdf")

plt.show()



#my_thresholds = [x / 10.0 for x in range(500, 450, -1)]
#y_class = (preds_prob2 > my_thresholds[20]).astype(int)



#preds_prob2[:, 1]
#fpr, tpr, thresh  = roc_curve(ytest_bin[:, 0], preds_prob2[:, 0])
#tpr[28]
#tpr[10]


#thresh = thresh[0: len(thresh) -30]
#fpr = fpr[0: len(fpr) -30]
#tpr = tpr[0: len(tpr) -30]
#plt.plot(fpr, tpr, color='darkorange',
#         lw=2)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(ytest_bin[:, i], preds_prob1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])






lw=2
cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = 50
#NUM_COLORS = 3
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(ytest.cat.categories[i], roc_auc[i]) )


# kaggle:
from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))




ytest.cat.categories[0]
ytest.cat.categories[1]
ytest.cat.categories[2]

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kaggle data, linear kernel')

plt.legend(loc="bottom")
plt.savefig(default_path + 'kaggle_lin2.pdf')
plt.show()







#plt.title('Kaggle data, semantic kernel with Wu-Palmer similarity')

thresh[10]
# linear model uses threshold thresh[10] = 48.31966631410343 for the decisions
decisions = (preds_prob2[:,0] >= thresh[10]).astype(int)
ytest_bin[0:49, 0] == decisions[0:49]

49/50

n_classes = ytest.cat.categories.shape[0]


ytest_bin = label_binarize(ytest, classes = ytest.cat.categories)
n_classes = ytest_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(ytest_bin[:, i], preds_prob1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ytest_bin.ravel(), preds_prob1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
clazz = 0
plt.plot(fpr[clazz], tpr[clazz], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[clazz])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(8,6), dpi=100)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = 50
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(ytest.cat.categories[i], roc_auc[i]) )



plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kaggle data, semantic kernel with Wu-Palmer similarity')

plt.legend(loc="lower right")
plt.show()

plt.savefig(default_path + 'kaggle_wu_micromacro.pdf')



